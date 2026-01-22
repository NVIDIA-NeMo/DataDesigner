# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from data_designer.config.seed import IndexRange, PartitionBlock


def test_index_range_validation():
    with pytest.raises(ValueError, match="should be greater than or equal to 0"):
        IndexRange(start=-1, end=10)

    with pytest.raises(ValueError, match="should be greater than or equal to 0"):
        IndexRange(start=0, end=-1)

    with pytest.raises(ValueError, match="'start' index must be less than or equal to 'end' index"):
        IndexRange(start=11, end=10)


def test_index_range_size():
    assert IndexRange(start=0, end=10).size == 11
    assert IndexRange(start=1, end=10).size == 10
    assert IndexRange(start=0, end=0).size == 1


def test_partition_block_validation():
    with pytest.raises(ValueError, match="should be greater than or equal to 0"):
        PartitionBlock(index=-1, num_partitions=10)

    with pytest.raises(ValueError, match="should be greater than or equal to 1"):
        PartitionBlock(index=0, num_partitions=0)

    with pytest.raises(ValueError, match="'index' must be less than 'num_partitions'"):
        PartitionBlock(index=10, num_partitions=10)


def test_partition_block_to_index_range():
    index_range = PartitionBlock(index=0, num_partitions=10).to_index_range(101)
    assert index_range.start == 0
    assert index_range.end == 9
    assert index_range.size == 10

    index_range = PartitionBlock(index=1, num_partitions=10).to_index_range(105)
    assert index_range.start == 10
    assert index_range.end == 19
    assert index_range.size == 10

    index_range = PartitionBlock(index=2, num_partitions=10).to_index_range(105)
    assert index_range.start == 20
    assert index_range.end == 29
    assert index_range.size == 10

    index_range = PartitionBlock(index=9, num_partitions=10).to_index_range(105)
    assert index_range.start == 90
    assert index_range.end == 104
    assert index_range.size == 15
