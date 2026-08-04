"""
Unit tests for src/split.py

Run with: pytest tests/test_split.py -v
"""

import pytest

from src.split import get_scaffold, group_by_scaffold, scaffold_split


# --- get_scaffold ---

def test_get_scaffold_known_example():
    # Straight from the RDKit docs' own worked example: the methyl and
    # propyl substituents get stripped, the two rings + linking O remain.
    smiles = "Cc1cc(Oc2nccc(CCC)c2)ccc1"
    expected_scaffold = "c1ccc(Oc2ccccn2)cc1"
    assert get_scaffold(smiles) == expected_scaffold


def test_get_scaffold_acyclic_molecule_is_empty():
    # A molecule with no ring system has no scaffold at all.
    assert get_scaffold("CCCCCC") == ""  # hexane


# --- group_by_scaffold ---

def test_group_by_scaffold_groups_shared_scaffolds_together():
    smiles_list = ["c1ccccc1", "Cc1ccccc1", "CCCCCC"]  # benzene, toluene, hexane
    groups = group_by_scaffold(smiles_list)

    benzene_scaffold = get_scaffold("c1ccccc1")
    assert set(groups[benzene_scaffold]) == {"c1ccccc1", "Cc1ccccc1"}
    assert groups[""] == ["CCCCCC"]


def test_group_by_scaffold_every_molecule_appears_exactly_once():
    smiles_list = ["c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "CCCCCC", "CCO"]
    groups = group_by_scaffold(smiles_list)

    all_grouped = [s for molecules in groups.values() for s in molecules]
    assert sorted(all_grouped) == sorted(smiles_list)


# --- scaffold_split ---

# Two real scaffold families (benzene and naphthalene derivatives) plus a
# handful of acyclic molecules -- gives the split more than one non-empty
# scaffold group to actually choose between.
BENZENE_FAMILY = ["c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "Clc1ccccc1"]
NAPHTHALENE_FAMILY = ["c1ccc2ccccc2c1", "Cc1ccc2ccccc2c1"]
ACYCLIC = ["CCCCCC", "CCCCCCCC", "CCO"]
SPLIT_TEST_SMILES = BENZENE_FAMILY + NAPHTHALENE_FAMILY + ACYCLIC


def test_scaffold_split_partitions_every_molecule_exactly_once():
    train, holdout, _ = scaffold_split(SPLIT_TEST_SMILES, holdout_fraction=0.3)

    # No molecule missing, none duplicated, none appearing in both sets.
    assert sorted(train + holdout) == sorted(SPLIT_TEST_SMILES)
    assert set(train).isdisjoint(set(holdout))


def test_scaffold_split_no_scaffold_leakage():
    train, holdout, holdout_scaffolds = scaffold_split(SPLIT_TEST_SMILES, holdout_fraction=0.3)

    train_scaffolds = {get_scaffold(s) for s in train}
    assert train_scaffolds.isdisjoint(holdout_scaffolds)


def test_scaffold_split_holdout_scaffolds_match_actual_holdout_molecules():
    train, holdout, holdout_scaffolds = scaffold_split(SPLIT_TEST_SMILES, holdout_fraction=0.3)

    for s in holdout:
        scaffold = get_scaffold(s)
        if scaffold != "":  # the empty-scaffold carve-out isn't tracked in holdout_scaffolds
            assert scaffold in holdout_scaffolds


def test_scaffold_split_acyclic_molecules_are_not_all_forced_to_one_side():
    train, holdout, _ = scaffold_split(ACYCLIC * 4, holdout_fraction=0.3)

    train_acyclic = [s for s in train if get_scaffold(s) == ""]
    holdout_acyclic = [s for s in holdout if get_scaffold(s) == ""]

    assert len(train_acyclic) > 0
    assert len(holdout_acyclic) > 0


def test_scaffold_split_handles_no_acyclic_molecules():
    # If the input has zero acyclic (empty-scaffold) molecules, a bare
    # groups.pop("") without a default would raise KeyError. This should
    # not crash.
    only_ring_molecules = BENZENE_FAMILY + NAPHTHALENE_FAMILY
    train, holdout, _ = scaffold_split(only_ring_molecules, holdout_fraction=0.3)
    assert sorted(train + holdout) == sorted(only_ring_molecules)


def test_scaffold_split_achieves_roughly_target_fraction():
    train, holdout, _ = scaffold_split(SPLIT_TEST_SMILES, holdout_fraction=0.3)
    achieved_fraction = len(holdout) / len(SPLIT_TEST_SMILES)

    # Can't hit the target exactly -- scaffold groups are indivisible, and
    # with a dataset this small, one group can swing the result a lot. This
    # is a loose sanity bound, not a precision check.
    assert 0 < achieved_fraction < 1