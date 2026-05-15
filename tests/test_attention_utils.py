# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
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

"""Unit tests for ``cola_dlm.attention_utils`` (no GPU required)."""

from __future__ import annotations

import torch

from cola_dlm.attention_utils import (
    create_na_block_causal_mask,
    cu_seqlens,
    get_seqlen,
    max_seqlen,
)


def test_cu_seqlens_shape() -> None:
    seq_len = torch.tensor([3, 0, 5, 2], dtype=torch.long)
    cu = cu_seqlens(seq_len)
    assert cu.dtype == torch.int32
    # skip_empty=True drops the zero, so cumulative over [3, 5, 2] with leading 0.
    assert cu.tolist() == [0, 3, 8, 10]


def test_get_seqlen_and_max_seqlen() -> None:
    seq_shape = torch.tensor([[3], [5], [2]], dtype=torch.long)
    lens = get_seqlen(seq_shape)
    assert lens.tolist() == [3, 5, 2]
    assert max_seqlen(lens) == 5


def test_na_block_causal_mask_single_sample() -> None:
    block_size = 2
    txt_shape = torch.tensor([[4]], dtype=torch.long)
    txt_q_shape = torch.tensor([[4]], dtype=torch.long)
    mask = create_na_block_causal_mask(
        txt_shape=txt_shape,
        txt_q_shape=txt_q_shape,
        block_size=block_size,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert mask.shape == (1, 1, 4, 4)

    min_val = torch.finfo(torch.float32).min
    # Block layout for L=4, block_size=2: blocks [0,0,1,1].
    # Q-block b_q may attend to K-blocks b_k <= b_q.
    expected_allowed = torch.tensor(
        [
            [True, True, False, False],
            [True, True, False, False],
            [True, True, True, True],
            [True, True, True, True],
        ]
    )
    allowed = mask[0, 0] == 0
    assert torch.equal(allowed, expected_allowed)
    # Disallowed positions must carry the dtype minimum (used as additive mask before softmax).
    assert torch.all(mask[0, 0][~expected_allowed] == min_val)


def test_na_block_causal_mask_multi_sample_no_crossing() -> None:
    """Two samples of length 2 should never attend across samples."""

    block_size = 2
    txt_shape = torch.tensor([[2], [2]], dtype=torch.long)
    txt_q_shape = torch.tensor([[2], [2]], dtype=torch.long)
    mask = create_na_block_causal_mask(
        txt_shape=txt_shape,
        txt_q_shape=txt_q_shape,
        block_size=block_size,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert mask.shape == (1, 1, 4, 4)

    allowed = mask[0, 0] == 0
    # Sample 0 occupies rows/cols 0-1, sample 1 occupies rows/cols 2-3.
    # Within a sample both positions attend to both (single block of size 2).
    expected = torch.tensor(
        [
            [True, True, False, False],
            [True, True, False, False],
            [False, False, True, True],
            [False, False, True, True],
        ]
    )
    assert torch.equal(allowed, expected)


def test_na_block_causal_mask_suffix_q() -> None:
    """Q == block_size should align to the tail of K within each sample."""

    block_size = 2
    txt_shape = torch.tensor([[4]], dtype=torch.long)
    txt_q_shape = torch.tensor([[2]], dtype=torch.long)
    mask = create_na_block_causal_mask(
        txt_shape=txt_shape,
        txt_q_shape=txt_q_shape,
        block_size=block_size,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert mask.shape == (1, 1, 2, 4)

    allowed = mask[0, 0] == 0
    # The single Q block is the 2nd (last) block of K (positions 2-3).
    # Block-causal: it attends to blocks 0 and 1 -> all 4 K positions.
    assert torch.all(allowed)
