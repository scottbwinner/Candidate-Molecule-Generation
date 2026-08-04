"""
Unit tests for src/tokenizer.py

Run with: pytest tests/test_tokenizer.py -v
(or just `pytest` from the project root to run the whole suite)

Requires a conftest.py at the project root that adds the root to
sys.path, so `from src.tokenizer import ...` resolves correctly.
"""

import pytest

from src.tokenizer import (
    PAD_TOKEN,
    START_TOKEN,
    END_TOKEN,
    SPECIAL_TOKENS,
    char_tokenize,
    atom_tokenize,
    build_vocab,
    encode,
    decode,
)


# A small set of real molecules chosen to exercise different edge cases:
# - CCO: simple, no branches, no rings
# - CC(=O)Oc1ccccc1C(=O)O (aspirin): branches, an aromatic ring, ring-closure digits
# - CCCl: a two-letter halogen (Cl)
# - C[C@@H](N)C(=O)O: a bracket atom with chirality
TEST_SMILES = [
    "CCO",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CCCl",
    "C[C@@H](N)C(=O)O",
]

MAX_LEN = 50  # generous headroom above every string in TEST_SMILES + <START>/<END>


# --- Round-trip: encode then decode should always return the original string ---

@pytest.mark.parametrize("tokenize_fn", [char_tokenize, atom_tokenize])
@pytest.mark.parametrize("smiles", TEST_SMILES)
def test_encode_decode_roundtrip(tokenize_fn, smiles):
    token2idx, idx2token = build_vocab(TEST_SMILES, tokenize_fn)
    indices = encode(smiles, tokenize_fn, token2idx, MAX_LEN)
    assert indices is not None
    assert decode(indices, idx2token) == smiles


# --- atom_tokenize: check the ACTUAL tokens produced, not just round-trip ---

def test_atom_tokenize_two_letter_halogen():
    assert atom_tokenize("CCCl") == ["C", "C", "Cl"]


def test_atom_tokenize_bracket_atom_is_one_token():
    assert atom_tokenize("C[C@@H](N)C(=O)O") == [
        "C", "[C@@H]", "(", "N", ")", "C", "(", "=", "O", ")", "O",
    ]


def test_atom_tokenize_preserves_full_string():
    # atom_tokenize already asserts this internally -- re-assert it here
    # explicitly so a regression shows up as a clearly named test failure,
    # not just a stack trace from inside the function under test.
    for smiles in TEST_SMILES:
        assert "".join(atom_tokenize(smiles)) == smiles


# --- char_tokenize: confirm it does NOT special-case multi-character atoms ---

def test_char_tokenize_splits_two_letter_atoms():
    assert char_tokenize("CCCl") == ["C", "C", "C", "l"]


# --- build_vocab: determinism (guards against the set-ordering bug) ---

def test_build_vocab_is_deterministic():
    token2idx_a, _ = build_vocab(TEST_SMILES, atom_tokenize)
    token2idx_b, _ = build_vocab(TEST_SMILES, atom_tokenize)
    assert token2idx_a == token2idx_b


# --- build_vocab: special token placement ---

def test_build_vocab_special_tokens_first():
    token2idx, idx2token = build_vocab(TEST_SMILES, atom_tokenize)

    assert token2idx[PAD_TOKEN] == 0
    for expected_idx, token in enumerate(SPECIAL_TOKENS):
        assert token2idx[token] == expected_idx

    # idx2token should be the exact inverse of token2idx
    assert all(idx2token[idx] == token for token, idx in token2idx.items())


def test_build_vocab_indices_are_contiguous_no_duplicates():
    token2idx, _ = build_vocab(TEST_SMILES, atom_tokenize)
    indices = sorted(token2idx.values())
    assert indices == list(range(len(token2idx)))


# --- encode: length handling ---

def test_encode_pads_short_sequences():
    token2idx, idx2token = build_vocab(TEST_SMILES, atom_tokenize)
    indices = encode("CCO", atom_tokenize, token2idx, max_len=10)
    assert indices is not None
    assert len(indices) == 10
    # <START>, C, C, O, <END> = 5 real tokens; the remaining 5 should be <PAD>
    assert indices[5:] == [token2idx[PAD_TOKEN]] * 5


def test_encode_exact_fit_no_padding():
    token2idx, idx2token = build_vocab(TEST_SMILES, atom_tokenize)
    exact_len = len(atom_tokenize("CCO")) + 2  # + <START> + <END>, no padding needed
    indices = encode("CCO", atom_tokenize, token2idx, max_len=exact_len)
    assert indices is not None
    assert len(indices) == exact_len
    assert idx2token[indices[-1]] == END_TOKEN  # last slot is <END>, not <PAD>


def test_encode_too_long_returns_none():
    token2idx, _ = build_vocab(TEST_SMILES, atom_tokenize)
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    too_short_max_len = len(atom_tokenize(aspirin))  # no room for <START>/<END>
    assert encode(aspirin, atom_tokenize, token2idx, too_short_max_len) is None


# --- decode: the no-<END> fallback (generation may run out of room) ---

def test_decode_missing_end_token_does_not_crash():
    token2idx, idx2token = build_vocab(TEST_SMILES, atom_tokenize)
    raw_tokens = [START_TOKEN, "C", "C", "O"]  # no <END> -- simulates max_len cutoff
    indices = [token2idx[t] for t in raw_tokens]

    assert decode(indices, idx2token) == "CCO"


def test_decode_strips_all_padding_after_end():
    token2idx, idx2token = build_vocab(TEST_SMILES, atom_tokenize)
    raw_tokens = [START_TOKEN, "C", "C", "O", END_TOKEN, PAD_TOKEN, PAD_TOKEN]
    indices = [token2idx[t] for t in raw_tokens]

    assert decode(indices, idx2token) == "CCO"