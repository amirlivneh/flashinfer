import pytest
import torch

import flashinfer
from flashinfer.utils import get_compute_capability


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cub_top_k_replays_independent_graphs_on_different_streams(monkeypatch):
    if get_compute_capability(torch.device("cuda"))[0] < 9:
        pytest.skip("large CUB segments require SM90+")
    monkeypatch.setenv("FLASHINFER_TOPK_ALGO", "cub")
    k = 256
    row = torch.randperm(32768, device="cuda").float()
    inputs = [row.expand(8, -1).contiguous(), row.roll(123).expand(8, -1).contiguous()]
    references = [torch.topk(scores, k) for scores in inputs]

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        flashinfer.top_k(inputs[0], k)
    capture_stream.synchronize()

    graphs, outputs = [], []
    for scores in inputs:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            output = flashinfer.top_k(scores, k)
        graphs.append(graph)
        outputs.append(output)

    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for graph, stream in zip(graphs, streams, strict=True):
        with torch.cuda.stream(stream):
            graph.replay()
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)

    for (values, indices), reference in zip(outputs, references, strict=True):
        sorted_values, order = values.sort(dim=-1, descending=True)
        torch.testing.assert_close(sorted_values, reference.values)
        torch.testing.assert_close(indices.gather(-1, order), reference.indices)
