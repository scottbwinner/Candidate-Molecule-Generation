"""
Unit tests for src/dataset.py

Run with: pytest tests/test_dataset.py -v
"""

import pytest
import torch
from torch.utils.data import DataLoader

from src.dataset import Zinc250kDataset


def make_tensor(rows):
    """Helper: build a small (N, max_len) long tensor from a list of lists."""
    return torch.tensor(rows, dtype=torch.long)


def test_len_matches_number_of_rows():
    dataset = Zinc250kDataset(make_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    assert len(dataset) == 3


def test_len_reflects_actual_tensor_size_not_hardcoded():
    # Two different sizes should give two different lengths -- guards
    # against a __len__ that happens to return the right number by luck
    # (e.g. a hardcoded constant) rather than actually reading the tensor.
    small = Zinc250kDataset(make_tensor([[1, 2], [3, 4]]))
    large = Zinc250kDataset(make_tensor([[1, 2]] * 20))
    assert len(small) == 2
    assert len(large) == 20


def test_getitem_returns_correct_row():
    dataset = Zinc250kDataset(make_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    assert torch.equal(dataset[0], torch.tensor([1, 2, 3], dtype=torch.long))
    assert torch.equal(dataset[1], torch.tensor([4, 5, 6], dtype=torch.long))
    assert torch.equal(dataset[2], torch.tensor([7, 8, 9], dtype=torch.long))


def test_getitem_returns_correct_shape():
    dataset = Zinc250kDataset(make_tensor([[1, 2, 3, 4, 5]] * 10))  # max_len = 5
    assert dataset[0].shape == (5,)


def test_getitem_returns_correct_dtype():
    dataset = Zinc250kDataset(make_tensor([[1, 2, 3]]))
    assert dataset[0].dtype == torch.long


def test_dataset_batches_correctly_through_dataloader():
    # Integration-style check: this is the whole point of building the
    # class this way, so worth confirming it actually works end to end,
    # not just that __len__/__getitem__ are individually correct.
    tensor = make_tensor([[i, i + 1, i + 2] for i in range(10)])
    dataset = Zinc250kDataset(tensor)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    batch = next(iter(loader))
    assert batch.shape == (4, 3)
    assert batch.dtype == torch.long
    assert torch.equal(batch[0], torch.tensor([0, 1, 2], dtype=torch.long))