"""
Unit tests for train_one_epoch() and evaluate_loss() in src/train.py.

main() is deliberately NOT covered here -- it's orchestration with real
side effects (file I/O, hardware detection, checkpoint saving), which is
better verified with an actual tiny-subset debug run than a unit test.
These two functions are pure enough (given a model/loader/optimizer/
criterion) to test properly in isolation, same as everything else in
this project.

Run with: pytest tests/test_train.py -v
"""

import math

import torch
from torch.utils.data import DataLoader

from src.dataset import Zinc250kDataset
from src.model import LSTMModel
from src.train import train_one_epoch, evaluate_loss


# Small dimensions -- fast to construct and run.
VOCAB_SIZE = 10
EMBED_DIM = 8
HIDDEN_DIM = 16
NUM_LAYERS = 1
PAD_IDX = 0
BATCH_SIZE = 4
SEQ_LEN = 6
DEVICE = torch.device("cpu")


def make_model():
    return LSTMModel(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        pad_idx=PAD_IDX,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=0.0,
    )


def make_loader(num_sequences=8, seed=0):
    torch.manual_seed(seed)
    # Values start at 1, not 0, so PAD_IDX never shows up in the middle of
    # a "real" sequence here -- these tests aren't about padding behavior.
    data = torch.randint(1, VOCAB_SIZE, (num_sequences, SEQ_LEN))
    dataset = Zinc250kDataset(data)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def make_criterion():
    return torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)


# --- evaluate_loss: must NOT change the model at all ---

def test_evaluate_loss_does_not_update_weights():
    model = make_model()
    loader = make_loader()
    criterion = make_criterion()

    params_before = [p.clone() for p in model.parameters()]
    evaluate_loss(model, loader, criterion, DEVICE)
    params_after = list(model.parameters())

    for before, after in zip(params_before, params_after):
        assert torch.equal(before, after)


def test_evaluate_loss_leaves_model_in_eval_mode():
    model = make_model()
    loader = make_loader()
    criterion = make_criterion()

    evaluate_loss(model, loader, criterion, DEVICE)
    assert model.training is False


def test_evaluate_loss_returns_valid_value():
    model = make_model()
    loader = make_loader()
    criterion = make_criterion()

    avg_loss = evaluate_loss(model, loader, criterion, DEVICE)

    assert isinstance(avg_loss, float)
    assert avg_loss >= 0
    assert math.isfinite(avg_loss)


# --- train_one_epoch: MUST change the model ---

def test_train_one_epoch_updates_weights():
    model = make_model()
    loader = make_loader()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = make_criterion()

    params_before = [p.clone() for p in model.parameters()]
    train_one_epoch(model, loader, optimizer, criterion, DEVICE)
    params_after = list(model.parameters())

    changed = any(
        not torch.equal(before, after)
        for before, after in zip(params_before, params_after)
    )
    assert changed


def test_train_one_epoch_leaves_model_in_train_mode():
    model = make_model()
    loader = make_loader()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = make_criterion()

    train_one_epoch(model, loader, optimizer, criterion, DEVICE)
    assert model.training is True


def test_train_one_epoch_returns_valid_values():
    model = make_model()
    loader = make_loader()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = make_criterion()

    avg_loss, clip_rate = train_one_epoch(model, loader, optimizer, criterion, DEVICE)

    assert isinstance(avg_loss, float)
    assert avg_loss >= 0
    assert math.isfinite(avg_loss)
    assert 0.0 <= clip_rate <= 1.0


# --- The main test: can the loop actually learn anything? ---

def test_train_one_epoch_reduces_loss_on_memorizable_data():
    # A couple of sequences repeated many times -- trivially memorizable.
    # If the loop is wired correctly end to end (shift-by-one split,
    # zero_grad placement, loss computed against the right target, the
    # optimizer step actually applied), loss should drop substantially
    # within a handful of epochs on data this easy.
    torch.manual_seed(0)
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = make_criterion()

    base_sequences = torch.randint(1, VOCAB_SIZE, (2, SEQ_LEN))
    data = base_sequences.repeat(20, 1)  # 40 rows, only 2 distinct sequences
    loader = DataLoader(Zinc250kDataset(data), batch_size=8, shuffle=True)

    first_epoch_loss, _ = train_one_epoch(model, loader, optimizer, criterion, DEVICE)

    final_loss = first_epoch_loss
    for _ in range(19):
        final_loss, _ = train_one_epoch(model, loader, optimizer, criterion, DEVICE)

    # Loose sanity bound, not a precision benchmark -- the point is
    # confirming real, substantial learning happened, not hitting an
    # exact number.
    assert final_loss < first_epoch_loss * 0.6