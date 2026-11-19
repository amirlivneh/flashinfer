from types import SimpleNamespace

import pytest
import torch

import flashinfer.topk as topk
from flashinfer import utils


@pytest.mark.cpu_only
@pytest.mark.parametrize("execution", ["eager", "graph"])
@pytest.mark.parametrize(
    ("entry", "workspace_arg"),
    [
        ("cub_topk", 3),
        ("cub_topk_page_table_transform", 8),
        ("cub_topk_ragged_transform", 4),
    ],
)
def test_cub_topk_workspace_is_private_when_calls_can_overlap(
    monkeypatch, execution, entry, workspace_arg
):
    # Use real Torch storage and FlashInfer's public dispatch and allocator.
    # Emulate CUDA stream identity, capture, and the native dispatch boundary.
    monkeypatch.setenv("FLASHINFER_TOPK_ALGO", "cub")
    monkeypatch.setattr(topk, "get_compute_capability", lambda _device: (9, 0))
    monkeypatch.setattr(utils, "_cache_buf", {})
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: execution == "graph"
    )
    stream = 1
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda *_a: SimpleNamespace(cuda_stream=stream)
    )

    workspaces = []

    def enqueue(*args):
        workspaces.append(args[workspace_arg])

    module = SimpleNamespace(
        **{entry: enqueue, f"{entry}_workspace_size": lambda *_args: 4096}
    )
    monkeypatch.setattr(topk, "get_topk_module", lambda: module)

    scores = torch.arange(32768, dtype=torch.float32).reshape(1, -1)
    lengths = torch.tensor([scores.size(1)], dtype=torch.int32)
    page_table = torch.arange(scores.size(1), dtype=torch.int32).reshape(1, -1)
    offsets = torch.zeros(1, dtype=torch.int32)

    def launch():
        if entry == "cub_topk_page_table_transform":
            topk.top_k_page_table_transform(scores, page_table, lengths, 128)
        elif entry == "cub_topk_ragged_transform":
            topk.top_k_ragged_transform(scores, offsets, lengths, 128)
        else:
            topk.top_k(scores, 128)

    launch()
    if execution == "eager":
        stream = 2
    # Two graphs captured on the same stream can replay on different streams.
    launch()

    first, second = workspaces
    # None asks the native binding to allocate private stream-ordered scratch.
    if first is not None and second is not None:
        assert first.data_ptr() != second.data_ptr(), (
            f"{entry} handed the same writable scratch to two {execution} calls"
        )
