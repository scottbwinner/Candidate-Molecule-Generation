from typing import Set, Dict, List, Tuple
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
import random

def get_scaffold(smiles: str) -> str:
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles)
    return scaffold


def group_by_scaffold(smiles_list: List[str]) -> Dict[str, List[str]]:
    groups = {}
    for smiles in smiles_list:
        scaffold = get_scaffold(smiles=smiles)
        if scaffold in groups:
            groups[scaffold].append(smiles)
        else:
            groups[scaffold] = [smiles]

    return groups


def scaffold_split(smiles_list: List[str], holdout_fraction: float, random_state: int = 6) -> Tuple[List[str], List[str], Set[str]]:
    """
    This function allows us to split our smiles data into training and holdout sets, while ensuring that similar molecules stay inside the same set.
    This allows us to guarantee that if a scaffold shows up in our holdout set, the model so no examples of that structural family during training, showing whether or not
    out model actually generates novel scaffolds.
    """
    groups = group_by_scaffold(smiles_list)

    # Molecules that have no rings all have an empty string as a scaffold. This doesn't make them similar, so when splitting between train 
    # and holdout, we split them normally.

    no_rings = groups.pop("", [])
    print("No rings: ", len(no_rings))
    train_no_rings, holdout_no_rings = train_test_split(no_rings, test_size=holdout_fraction, random_state=random_state)


    # All other molecules need to be split such that they stay in the same set with their scaffold.
    
    scaffolds = list(groups.keys())
    random.shuffle(scaffolds, random_state=random_state)

    total_molecules = sum(len(v) for v in groups.values())
    holdout_count = 0
    holdout = []
    holdout_scaffolds = set()
    for scaffold in scaffolds:
        molecules = groups.pop(scaffold)
        holdout.extend(molecules)
        holdout_count += len(molecules)
        holdout_scaffolds.add(scaffold)
        if holdout_count >= holdout_fraction * total_molecules:
            break
    holdout_set = holdout + holdout_no_rings
    training_set = [item for sublist in groups.values() for item in sublist] + train_no_rings

    print("Actual Holdout %: ", len(holdout_set) / len(smiles_list))

    return (training_set, holdout_set, holdout_scaffolds)





