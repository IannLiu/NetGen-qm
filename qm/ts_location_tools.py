from typing import List, Literal
from rdkit import Chem
from scipy.optimize import minimize
import numpy as np

import autode as ade
from autode.substitution import get_substc_and_add_dummy_atoms, get_cost_rotate_translate
from autode.species import Complex
from autode.bond_rearrangement import BondRearrangement

from rdkit_util import str_to_mol, drop_map_num


def match_atom_nums(smi1: str, smi2: str):
    """
    Matching atom map number {number of smi1: number of smi2}

    smi1: atom-mapped smi1
    smi2: atom-mapped smi2
    Returns: dict

    """
    assert drop_map_num(smi1) == drop_map_num(smi2)
    mol1, mol2 = str_to_mol(smi1), str_to_mol(smi2)
    matched_atom_idx = mol1.GetSubstructMatch(mol2)
    matched_atoms = [mol2.GetAtomWithIdx(idx) for idx in range(len(matched_atom_idx))]
    source_atoms = [mol1.GetAtomWithIdx(idx) for idx in matched_atom_idx]
    old_new_idx = {s.GetAtomMapNum(): m.GetAtomMapNum() for s, m in zip(source_atoms, matched_atoms)}
    return old_new_idx


def relable_atom_nums(rsmiles_list: List[str], psmiles_list: List[str]):
    """
    Relable atom map number from 0 to number_of_atoms -1
    """
    ratom_nums = [atom.GetAtomMapNum() for smi in rsmiles_list for atom in str_to_mol(smi).GetAtoms()]
    patom_nums = [atom.GetAtomMapNum() for smi in psmiles_list for atom in str_to_mol(smi).GetAtoms()]
    assert sorted(ratom_nums) == sorted(patom_nums), "The atom mapping between the reactants and products doesn't match"
    reordered_num_mapping = {num: idx for idx, num in enumerate(ratom_nums)}
    rmols = [str_to_mol(smi) for smi in rsmiles_list]
    for mol in rmols:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(reordered_num_mapping[atom.GetAtomMapNum()])
    reordered_rsmis_list = [Chem.MolToSmiles(mol) for mol in rmols]

    pmols = [str_to_mol(smi) for smi in psmiles_list]
    for mol in pmols:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(reordered_num_mapping[atom.GetAtomMapNum()])
    reordered_psmis_list = [Chem.MolToSmiles(mol) for mol in pmols]

    return reordered_rsmis_list,  reordered_psmis_list


def initialize_autode_complex(smiles_list: List[str], label: Literal['reacs', 'prods'], ade_obj: ade):
    unmapped_smi_list = [drop_map_num(smi) for smi in smiles_list]
    # initialize autode smiles
    atom_numbers = []
    mapped_smiles_list = []
    for ith, smi in enumerate(unmapped_smi_list):
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        atom_numbers.append(mol.GetNumAtoms())
        start_num = atom_numbers[ith - 1] if ith > 0 else 0
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + start_num)
        mapped_smiles_list.append(Chem.MolToSmiles(mol))

    # initialize autode molecules
    input_list = [ade_obj.Molecule(name=f'reactant{idx}', smiles=smi) for idx, smi in enumerate(unmapped_smi_list)]

    for mol in input_list:
        try:
            mol.find_lowest_energy_conformer()
        except:
            continue

    if label == 'reacs':
        complex_mols = ade_obj.species.ReactantComplex(*input_list, name='reactants')
    else:
        complex_mols = ade_obj.species.ProductComplex(*input_list, name='products', )

    return complex_mols, mapped_smiles_list


def translate_rotate_reactant(
    reactant, bond_rearrangement, shift_factor, n_iters=10
):
    """
    Shift a molecule in the reactant complex so that the attacking atoms
    (a_atoms) are pointing towards the attacked atoms (l_atoms). Applied in
    place

    ---------------------------------------------------------------------------
    Arguments:
        reactant (autode.complex.Complex):

        bond_rearrangement (autode.bond_rearrangement.BondRearrangement):

        shift_factor (float):

        n_iters (int): Number of iterations of translation/rotation to perform
                       to (hopefully) find the global minima
    """

    if not isinstance(reactant, Complex):
        return

    if reactant.n_molecules < 2:
        return

    # This function can add dummy atoms for e.g. SN2' reactions where there
    # is not a A -- C -- Xattern for the substitution centre
    subst_centres = get_substc_and_add_dummy_atoms(
        reactant, bond_rearrangement, shift_factor=shift_factor
    )

    if all(
        sc.a_atom in reactant.atom_indexes(mol_index=0) for sc in subst_centres
    ):
        attacking_mol = 0
    else:
        attacking_mol = 1

    # Find the global minimum for inplace rotation, translation and rotation
    min_cost, opt_x = None, None

    for _ in range(n_iters):
        res = minimize(
            get_cost_rotate_translate,
            x0=np.random.random(11),
            method="BFGS",
            tol=0.1,
            args=(reactant, subst_centres, attacking_mol),
        )

        if min_cost is None or res.fun < min_cost:
            min_cost = res.fun
            opt_x = res.x

    # Translate/rotation the attacking molecule optimally
    reactant.rotate_mol(
        axis=opt_x[:3], theta=opt_x[3], mol_index=attacking_mol
    )
    reactant.translate_mol(vec=opt_x[4:7], mol_index=attacking_mol)
    reactant.rotate_mol(
        axis=opt_x[7:10], theta=opt_x[10], mol_index=attacking_mol
    )

    reactant.atoms.remove_dummy()

    return reactant


def bond_rearrangement(forming_bonds: List[tuple], breaking_bonds: List[tuple]):
    """
    Return a BondRearrangement class of autode
    Args:
        forming_bonds: formation bonds
        breaking_bonds: broken bonds

    Returns:

    """
    return BondRearrangement(forming_bonds=forming_bonds, breaking_bonds=breaking_bonds)