# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in host check that GDN prefill ``checkpoint_cu_starts`` is in slots."""

import pytest
import torch

from flashinfer.gdn_prefill import _validate_checkpoint_cu_starts


def _i64(values):
    return torch.tensor(values, dtype=torch.int64)


def _check(cu_seqlens, starts, *, rows, every=64, **kwargs):
    _validate_checkpoint_cu_starts(
        cu_seqlens=_i64(cu_seqlens),
        checkpoint_cu_starts=_i64(starts),
        checkpoint_every_n_tokens=every,
        num_checkpoint_rows=rows,
        **kwargs,
    )


@pytest.fixture
def validate(monkeypatch):
    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("FLASHINFER_VALIDATE_INPUTS", raising=False)
    _check([0, 256, 512], [0, 256, 512], rows=8)


def test_slot_offsets_accepted(validate):
    _check([0, 256, 512], [0, 4, 8], rows=8)
    # Partial trailing intervals store no checkpoint.
    _check([0, 100, 300, 300], [0, 1, 4, 4], rows=4)


def test_token_offsets_rejected(validate):
    with pytest.raises(ValueError, match="not token offsets"):
        _check([0, 256, 512], [0, 256, 512], rows=8)


def test_too_few_rows_rejected(validate):
    with pytest.raises(ValueError, match="state_checkpoints has 7 rows"):
        _check([0, 256, 512], [0, 4, 8], rows=7)


def test_prefix_lens_counts_absolute_positions(validate):
    # Prefix 32 + 96 new tokens crosses positions 64 and 128: two checkpoints,
    # where 96 tokens without the prefix would store one.
    prefix_lens = _i64([32, 0])
    _check([0, 96, 160], [0, 2, 3], rows=3, prefix_lens=prefix_lens)
    with pytest.raises(ValueError, match="expected \\[0, 2, 3\\]"):
        _check([0, 96, 160], [0, 1, 2], rows=3, prefix_lens=prefix_lens)


def test_checkpoint_pool_bounds(validate):
    pool = _i64([5, 6, 7, 8])
    # Only per-sequence first entries are read; [num_seqs] entries suffice.
    _check([0, 128, 256], [0, 2], rows=16, checkpoint_state_indices=pool)
    with pytest.raises(ValueError, match="outside the 4 entries"):
        _check([0, 128, 256], [0, 128], rows=16, checkpoint_state_indices=pool)
