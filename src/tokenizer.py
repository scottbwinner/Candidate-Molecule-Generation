"""
SMILES tokenization: two interchangeable strategies (character-level and
atom-level), plus vocabulary construction and encode/decode built on top
of whichever one you pass in.

Design: char_tokenize and atom_tokenize share the signature
    (smiles: str) -> List[str]
build_vocab/encode/decode just take a tokenize_fn argument to determine the active strategy.
"""

import re
from typing import Callable, Dict, List, Tuple

# --- Special tokens ---

PAD_TOKEN = "<PAD>"
START_TOKEN = "<START>"
END_TOKEN = "<END>"
SPECIAL_TOKENS = [PAD_TOKEN, START_TOKEN, END_TOKEN]  # PAD first -> index 0

# --- Atom-level tokenization ---
# Schwaller et al.'s regex (Molecular Transformer paper)
SMI_REGEX_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"


# A tokenizer function turns one SMILES string into a list of token strings.
TokenizeFn = Callable[[str], List[str]]


def char_tokenize(smiles: str) -> List[str]:
    """
    Naive character-level tokenization: every character is its own token.

    Does not keep multi-character atoms (Cl, Br) or bracket expressions of atoms ([C@@H]) in their own tokens.
    """
    tokens = list(smiles)
    return tokens


def atom_tokenize(smiles: str) -> List[str]:
    """
    Atom-level tokenization using the Schwaller et al. regex: bracket
    expressions of atoms ([C@@H]) and two-letter halogens (Cl, Br) come out as single
    tokens instead of being split character-by-character.
    """
    tokens = re.findall(SMI_REGEX_PATTERN, smiles)
    assert "".join(tokens) == smiles
    return tokens


# --- Vocabulary ---

def build_vocab(
    smiles_list: List[str], tokenize_fn: TokenizeFn
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Scan every SMILES string in smiles_list and build the unique vocabulary of tokens.

    Returns (token2idx, idx2token), with SPECIAL_TOKENS assigned first
    (so PAD_TOKEN = index 0, matching nn.Embedding's padding_idx convention).
    """

    vocab = []

    for smile in smiles_list:
        smile_tokens = tokenize_fn(smile)
        vocab += smile_tokens

    vocab = sorted(set(vocab))

    token2idx = {}
    idx2token = {}

    for index, token in enumerate(SPECIAL_TOKENS):
        token2idx[token] = index
        idx2token[index] = token


    for token in vocab:
        idx = len(token2idx)
        token2idx[token] = idx
        idx2token[idx] = token

    return (token2idx, idx2token)


# --- Encode / decode ---

def encode(
    smiles: str,
    tokenize_fn: TokenizeFn,
    token2idx: Dict[str, int],
    max_len: int,
) -> List[int]:
    """
    Tokenizes and encodes SMILES string. 
    If SMILES string has more tokens than max_len, returns None.
    """

    tokens = tokenize_fn(smiles)
    tokens.insert(0, START_TOKEN)
    tokens.append(END_TOKEN)
    if len(tokens) > max_len:
        return None
    else:
        tokens.extend([PAD_TOKEN] * max(0, max_len - len(tokens)))

    indices = [token2idx[token] for token in tokens]
    return indices



def decode(indices: List[int], idx2token: Dict[int, str]) -> str:
    """
    Inverse of encode: map ids back to tokens, strips
    special tokens, join back into a single SMILES string.
    """

    tokens = [idx2token[index] for index in indices]

    tokens.remove(START_TOKEN)

    if END_TOKEN not in tokens:
            return "".join(tokens)
    
    tokens = tokens[0:tokens.index(END_TOKEN)]

    smiles = "".join(tokens)

    return smiles
