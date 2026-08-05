from rdkit import Chem
from rdkit.Chem import QED
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # src/train.py -> src/ -> project root
data_dir = PROJECT_ROOT / "data" / "processed" / "char_tokenized"

from src.split import get_scaffold

def compute_validity(smiles_list):
    invalid_count = 0
    valid_smiles = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_count += 1
        else:
            # Getting canonical_smiles this way guarantees that all equivalent molecules are expressed in only one way
            canonical_smiles = Chem.MolToSmiles(mol)
            valid_smiles.append(canonical_smiles)
    return valid_smiles, invalid_count


def compute_uniqueness(smiles_list):
    if len(smiles_list) == 0:
        return 0
    return len(set(smiles_list)) / len(smiles_list)


def compute_novelty(canonical_training_smiles, valid_generated_smiles):
    novel = []
    not_novel = []
    # Only get unique smiles
    valid_generated_smiles = set(valid_generated_smiles)
    for smiles in valid_generated_smiles:
        if smiles in canonical_training_smiles:
            not_novel.append(smiles)
        else:
            novel.append(smiles)

    novelty_rate = len(novel) / len(valid_generated_smiles)
    return novel, not_novel, novelty_rate


def compute_qed_scores(smiles_list):
    scores = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        scores.append(QED.qed(mol))
    return scores


def compute_scaffold_novelty(smiles_list, holdout_scaffolds):
    scaffold_novelty = []
    for smiles in smiles_list:
        scaffold = get_scaffold(smiles)
        if scaffold in holdout_scaffolds:
            scaffold_novelty.append(1)
        else:
            scaffold_novelty.append(0)
    if len(scaffold_novelty) == 0:
        return 0
    return sum(scaffold_novelty) / len(scaffold_novelty)