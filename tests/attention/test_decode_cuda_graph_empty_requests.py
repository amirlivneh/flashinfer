"""CUDA-graph decode planning with empty (0-page) requests.

PartitionPagedKVCacheBinarySearchMinNumPagePerBatch counted an empty request as 0
work items while choosing the split-KV chunk size, but DecodeSplitKVIndptr gives it
one. Under CUDA graphs DecodePlanImpl reserves only max_grid_size / gdy work slots,
so the plan wrote more work items than slots: the CTAs past the last slot never ran
and the merge read uninitialized partial outputs for the long request.
"""

import pytest
import torch

import flashinfer


def _graph_wrapper(batch_size, max_pages):
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    return flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        ws,
        "NHD",
        use_cuda_graph=True,
        use_tensor_cores=False,
        paged_kv_indptr_buffer=torch.empty(
            batch_size + 1, dtype=torch.int32, device="cuda"
        ),
        paged_kv_indices_buffer=torch.empty(
            max_pages, dtype=torch.int32, device="cuda"
        ),
        paged_kv_last_page_len_buffer=torch.empty(
            batch_size, dtype=torch.int32, device="cuda"
        ),
    )


@pytest.mark.parametrize("num_kv_heads", [1, 8])
def test_cuda_graph_decode_with_empty_requests(num_kv_heads):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    torch.manual_seed(0)
    page_size, head_dim, group = 16, 128, 4
    num_qo_heads = num_kv_heads * group
    min_chunk_pages = max(128 // page_size, 1)

    # Number of work slots the CUDA-graph plan reserves (max_grid_size / gdy).
    probe = _graph_wrapper(1, 1)
    probe.plan(
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        data_type=torch.float16,
    )
    slots = int(probe._plan_info[0])

    # slots - 1 requests keeps batch_size * gdy < max_grid_size (the split path).
    # One long request needs exactly `slots` chunks at the minimum chunk size, so a
    # search that ignores the empty requests stops there and leaves no slot for them.
    batch_size = slots - 1
    long_pages = slots * min_chunk_pages
    indptr = torch.zeros(batch_size + 1, dtype=torch.int32)
    indptr[-1] = long_pages
    indices = torch.arange(long_pages, dtype=torch.int32)
    last_page_len = torch.full((batch_size,), page_size, dtype=torch.int32)

    kv = torch.randn(
        long_pages,
        2,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    q = torch.randn(
        batch_size, num_qo_heads, head_dim, dtype=torch.float16, device="cuda"
    )

    graph = _graph_wrapper(batch_size, long_pages)
    graph.plan(
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        data_type=torch.float16,
    )
    out = graph.run(q, kv)

    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    eager = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
    eager.plan(
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        data_type=torch.float16,
    )
    ref = eager.run(q, kv)

    torch.testing.assert_close(out[-1], ref[-1], rtol=1e-3, atol=1e-3)
