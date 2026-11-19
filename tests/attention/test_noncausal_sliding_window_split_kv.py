"""FA2 batch prefill: non-causal sliding window under split-KV.

The planner bounded every request's split-KV span by ``window_left + cta_tile_q``,
and the kernel re-derived the chunk count from the same bound. With a causal mask a
query tile needs at most that many keys, but without one each query attends to
every key from its window start to ``kv_len``: the first tile needs
``qo_len + window_left`` keys. The chunks stopped short, so those keys were dropped.
"""

import math

import pytest
import torch

import flashinfer

PLAN_INFO_SPLIT_KV = 14  # PrefillPlanInfo::ToVector() index of split_kv


def _reference(q, k, v, window_left):
    # q: [qo_len, H, D], k/v: [kv_len, H, D]; non-causal left window:
    # key j is visible to query i iff j + qo_len + window_left >= kv_len + i.
    qo_len, kv_len = q.shape[0], k.shape[0]
    i = torch.arange(qo_len, device=q.device)[:, None]
    j = torch.arange(kv_len, device=q.device)[None, :]
    mask = j + qo_len + window_left >= kv_len + i
    s = torch.einsum("qhd,khd->hqk", q.float(), k.float()) / math.sqrt(q.shape[-1])
    s = s.masked_fill(~mask[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", torch.softmax(s, dim=-1), v.float()).half()


@pytest.mark.parametrize("layout", ["paged", "ragged"])
@pytest.mark.parametrize(
    "qo_len,kv_len,window_left", [(1024, 4096, 256), (517, 3001, 100)]
)
def test_noncausal_sliding_window_split_kv(layout, qo_len, kv_len, window_left):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    torch.manual_seed(0)
    dev = "cuda"
    page_size, num_heads, head_dim = 16, 1, 128
    q = torch.randn(qo_len, num_heads, head_dim, dtype=torch.half, device=dev)
    k = torch.randn(kv_len, num_heads, head_dim, dtype=torch.half, device=dev)
    v = torch.randn(kv_len, num_heads, head_dim, dtype=torch.half, device=dev)
    qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device=dev)
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)

    if layout == "paged":
        num_pages = (kv_len + page_size - 1) // page_size
        pad = num_pages * page_size - kv_len
        k_pages = torch.cat([k, k.new_zeros(pad, num_heads, head_dim)]).view(
            num_pages, page_size, num_heads, head_dim
        )
        v_pages = torch.cat([v, v.new_zeros(pad, num_heads, head_dim)]).view(
            num_pages, page_size, num_heads, head_dim
        )
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="fa2")
        w.plan(
            qo_indptr,
            torch.tensor([0, num_pages], dtype=torch.int32, device=dev),
            torch.arange(num_pages, dtype=torch.int32, device=dev),
            torch.tensor(
                [kv_len - (num_pages - 1) * page_size], dtype=torch.int32, device=dev
            ),
            num_heads,
            num_heads,
            head_dim,
            page_size,
            causal=False,
            window_left=window_left,
        )
        out = w.run(q, (k_pages, v_pages))
    else:
        w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend="fa2")
        w.plan(
            qo_indptr,
            torch.tensor([0, kv_len], dtype=torch.int32, device=dev),
            num_heads,
            num_heads,
            head_dim,
            causal=False,
            window_left=window_left,
        )
        out = w.run(q, k, v)

    # The bug only shows on the split-KV path; make sure this shape takes it.
    assert w._plan_info[PLAN_INFO_SPLIT_KV], (
        "shape no longer exercises split-KV; adjust it"
    )
    torch.testing.assert_close(
        out, _reference(q, k, v, window_left), rtol=2e-3, atol=2e-3
    )
