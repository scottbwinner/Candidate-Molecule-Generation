from typing import Set, Dict, List, Tuple
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
import random

def get_scaffold(smiles: str) -> str:
    """
    Molecular Representation can be quite difficult, because structurally similar molecules can look completely different as SMILES strings. 
    Many different slight alterations on the same molecule could result in many different representations of essentially
    the same molecule.

    To be able to properly identify similar molecules, we can calculate a molecule's 'scaffold', defined in Bemis & Murcko's paper 
    "The Propeties of Known Drugs. 1. Molecular Frameworks". A molecule's scaffold is essentially the backbone of the molecule,
    everything left after you strip away all the terminal side chains. After getting rid of the terminal side chains we are left 
    with a scaffold of the ring systems in the molecule and linker atoms connecting those rings together.

    If a molecule has no ring systems, it does not have a scaffold. In this case we will return None.

    This function takes in a SMILES string and returns the corresponding scaffold.

    Parameters:
        smiles: SMILES string

    Returns:
        scaffold: SMILES string's corresponding scaffold.
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles)
    return scaffold


def group_by_scaffold(smiles_list: List[str]) -> Dict[str, List[str]]:
    """
    This function takes in a list of SMILES strings, and groups the SMILES strings by their corresponding scaffoldings.

    Parameters:
        smiles_list: List of SMILES strings
    
    Returns:
        groups: Dictionary where each key is a scaffolding, and the corresponding value is a list of molecules in 
        smiles_list that has that scaffolding.
    """
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
    This function allows us to split our smiles data into training and holdout sets, while ensuring that similar molecules stay inside 
    the same set. This guarantees that no scaffold appears in both the training and holdout sets, so any evaluation performed
    against the holdout set reflects generalization to unseen molecule scaffolds.

    The edge case of this function is for molecules with no rings and therefore no scaffold. These molecules will be split normally, 
    since all molecules with no rings are not 'similar'.

    Parameters:
        smiles_list: List of SMILES strings
        holdout_fraction: Float for desired fraction of data that should be in our holdout set. 
                          Thus 1 - holdout_fraction is the desired fraction of data that should be in our training set.
        random_state: Random State

    Returns:
        (training_set, holdout_set, holdout_scaffolds)
        training_set: Training set (with no shared scaffolds with holdout set)
        holdout_set: Holdout set (with no shared scaffolds from training set)
        holdout_scaffolds: Molecules scaffolds that appear in the holdout set.
    """
    groups = group_by_scaffold(smiles_list)

    # Molecules that have no rings all have an empty string as a scaffold. This doesn't make them similar, so when splitting between train 
    # and holdout, we split them normally.

    no_rings = groups.pop("", [])
    print("No rings: ", len(no_rings))
    if len(no_rings) == 0:
        train_no_rings, holdout_no_rings = [], []
    else:
        train_no_rings, holdout_no_rings = train_test_split(no_rings, test_size=holdout_fraction, random_state=random_state)


    # All other molecules need to be split such that they stay in the same set with their scaffold.
    
    scaffolds = list(groups.keys())
    random.seed(random_state)
    random.shuffle(scaffolds)

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





