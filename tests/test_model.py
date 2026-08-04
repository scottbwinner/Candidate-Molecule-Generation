"""
Unit tests for src/model.py

Run with: pytest tests/test_model.py -v
"""

import warnings

import pytest
import torch

from src.model import LSTMModel


# Small dimensions -- fast to construct and run; the actual values don't
# matter for these tests since nothing here checks learned behavior.
VOCAB_SIZE = 12
EMBED_DIM = 8
HIDDEN_DIM = 16
NUM_LAYERS = 2
PAD_IDX = 0
DROPOUT = 0.2
BATCH_SIZE = 4
SEQ_LEN = 7


def make_model(num_layers=NUM_LAYERS, dropout=DROPOUT):
    return LSTMModel(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        pad_idx=PAD_IDX,
        hidden_dim=HIDDEN_DIM,
        num_layers=num_layers,
        dropout=dropout,
    )


# --- forward(): the core smoke test ---

def test_forward_output_shapes():
    model = make_model()
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    logits, (h_n, c_n) = model(x)

    assert logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    assert h_n.shape == (NUM_LAYERS, BATCH_SIZE, HIDDEN_DIM)
    assert c_n.shape == (NUM_LAYERS, BATCH_SIZE, HIDDEN_DIM)



def test_forward_accepts_and_threads_hidden_state():
    # Confirms the hidden state returned by one call can be fed into the
    # next call without crashing or changing shape -- exactly the pattern
    # sample.py's generation loop will rely on, one token at a time.
    model = make_model()
    x1 = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 1))
    x2 = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 1))

    _, hidden1 = model(x1)
    logits2, hidden2 = model(x2, hidden1)

    assert logits2.shape == (BATCH_SIZE, 1, VOCAB_SIZE)
    assert hidden2[0].shape == hidden1[0].shape
    assert hidden2[1].shape == hidden1[1].shape


# --- init_hidden() ---

def test_init_hidden_shape_and_zeros():
    model = make_model()
    h0, c0 = model.init_hidden(batch_size=BATCH_SIZE, device=torch.device("cpu"))

    assert h0.shape == (NUM_LAYERS, BATCH_SIZE, HIDDEN_DIM)
    assert c0.shape == (NUM_LAYERS, BATCH_SIZE, HIDDEN_DIM)
    assert torch.all(h0 == 0)
    assert torch.all(c0 == 0)


def test_init_hidden_respects_requested_batch_size():
    # Guards against batch_size being silently hardcoded anywhere.
    model = make_model()
    h0, _ = model.init_hidden(batch_size=1, device=torch.device("cpu"))
    assert h0.shape == (NUM_LAYERS, 1, HIDDEN_DIM)


# --- padding_idx wiring ---

def test_padding_idx_embedding_row_is_zero():
    # nn.Embedding(padding_idx=...) initializes that row to all zeros --
    # a quick check that PAD_IDX actually reached the embedding layer
    # rather than being silently dropped somewhere in __init__.
    model = make_model()
    assert torch.all(model.embedding.weight[PAD_IDX] == 0)


# --- dropout guard: num_layers=1 must not pass nonzero dropout to nn.LSTM ---

def test_single_layer_with_dropout_does_not_warn():
    # nn.LSTM warns if dropout > 0 is passed with num_layers=1, since there's
    # no "between layers" for it to apply to. This should never happen here
    # if the `dropout if num_layers > 1 else 0` guard is actually in place.
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # turn any warning into a failure
        make_model(num_layers=1, dropout=0.5)


# --- gradient flow ---

def test_gradients_flow_to_embedding_layer():
    # Correct output shape doesn't guarantee the model can actually learn --
    # this confirms backprop genuinely reaches the embedding weights, which
    # a shape-only smoke test would not catch (e.g. an errant .detach()
    # or a stray no_grad() block cutting the graph somewhere).
    model = make_model()
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    target = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    logits, _ = model(x)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE), target.reshape(-1)
    )
    loss.backward()

    assert model.embedding.weight.grad is not None
    assert not torch.all(model.embedding.weight.grad == 0)