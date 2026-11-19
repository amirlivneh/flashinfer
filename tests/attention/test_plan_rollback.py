"""A plan() that raises must leave the wrapper on its previous plan.

plan() copies the new batch into buffers that run() and captured CUDA graphs read,
and only afterwards finishes validating it. A rejection used to leave the old
schedule (plan_info) over the new batch's buffers.

The CPU tests need no kernel: the wrappers are built on CPU tensors and the rejected
plan() raises before any CUDA work. The GPU test checks run() after a rejection.
"""

import pytest
import torch

import flashinfer


@pytest.fixture
def no_pinned_memory(monkeypatch):
    """Let the wrappers' pinned host mirrors be ordinary memory without CUDA."""
    if torch.cuda.is_available():
        return
    real_empty = torch.empty

    def empty(*args, pin_memory=False, **kwargs):
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)


def _decode_graph_wrapper(device, batch, max_pages):
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    return flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        ws,
        "NHD",
        use_cuda_graph=True,
        use_tensor_cores=False,
        paged_kv_indptr_buffer=torch.zeros(batch + 1, dtype=torch.int32, device=device),
        paged_kv_indices_buffer=torch.zeros(
            max_pages, dtype=torch.int32, device=device
        ),
        paged_kv_last_page_len_buffer=torch.zeros(
            batch, dtype=torch.int32, device=device
        ),
    )


def _snapshot(w, names):
    return {n: getattr(w, n).clone() for n in names}


def test_rejected_decode_plan_keeps_graph_buffers(no_pinned_memory):
    batch = 4
    w = _decode_graph_wrapper("cpu", batch, 16)
    names = [
        "_paged_kv_indptr_buf",
        "_paged_kv_indices_buf",
        "_paged_kv_last_page_len_buf",
    ]
    before = _snapshot(w, names)
    state_before = dict(w.__dict__)

    indptr = torch.arange(0, 4 * (batch + 1), 4, dtype=torch.int32)
    indices = torch.arange(4 * batch, dtype=torch.int32).flip(0)
    last = torch.full((batch,), 7, dtype=torch.int32)
    # fixed_split_size is tensor-core only: plan() rejects it after the copies.
    with pytest.raises(ValueError, match="fixed_split_size"):
        w.plan(indptr, indices, last, 32, 8, 128, 16, fixed_split_size=64)

    for n in names:
        assert torch.equal(getattr(w, n), before[n]), f"rejected plan() rewrote {n}"
    assert w.__dict__.keys() == state_before.keys()
    for k, v in state_before.items():
        assert w.__dict__[k] is v, f"rejected plan() changed wrapper state {k}"


def test_rejected_paged_prefill_plan_keeps_graph_buffers(no_pinned_memory):
    batch, max_pages = 2, 8
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8)
    bufs = dict(
        qo_indptr_buf=torch.zeros(batch + 1, dtype=torch.int32),
        paged_kv_indptr_buf=torch.zeros(batch + 1, dtype=torch.int32),
        paged_kv_indices_buf=torch.zeros(max_pages, dtype=torch.int32),
        paged_kv_last_page_len_buf=torch.zeros(batch, dtype=torch.int32),
    )
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, "NHD", use_cuda_graph=True, backend="fa2", **bufs
    )
    before = {k: v.clone() for k, v in bufs.items()}
    # A custom mask with no custom_mask_buf is rejected after the copies.
    with pytest.raises(ValueError, match="custom_mask_buf"):
        w.plan(
            torch.tensor([0, 2, 4], dtype=torch.int32),
            torch.tensor([0, 1, 2], dtype=torch.int32),
            torch.tensor([3, 5], dtype=torch.int32),
            torch.tensor([4, 4], dtype=torch.int32),
            8,
            8,
            128,
            16,
            packed_custom_mask=torch.zeros(16, dtype=torch.uint8),
        )
    for k, v in bufs.items():
        assert torch.equal(v, before[k]), f"rejected plan() rewrote {k}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_run_after_rejected_decode_plan_still_runs_previous_plan():
    torch.manual_seed(0)
    dev = "cuda"
    batch, pages, page_size, H, Hkv, D = 4, 4, 16, 32, 8, 128
    w = _decode_graph_wrapper(dev, batch, batch * pages)
    kv = torch.randn(batch * pages, 2, page_size, Hkv, D, dtype=torch.half, device=dev)
    q = torch.randn(batch, H, D, dtype=torch.half, device=dev)

    w.plan(
        torch.arange(0, pages * (batch + 1), pages, dtype=torch.int32),
        torch.arange(batch * pages, dtype=torch.int32),
        torch.full((batch,), page_size, dtype=torch.int32),
        H,
        Hkv,
        D,
        page_size,
        data_type=torch.half,
    )
    ref = w.run(q, kv)

    with pytest.raises(ValueError):
        w.plan(
            torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),
            torch.tensor([15, 14, 13, 12], dtype=torch.int32),
            torch.ones(batch, dtype=torch.int32),
            H,
            Hkv,
            D,
            page_size,
            data_type=torch.half,
            fixed_split_size=64,
        )
    torch.testing.assert_close(w.run(q, kv), ref, rtol=1e-3, atol=1e-3)
