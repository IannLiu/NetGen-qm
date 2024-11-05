from rdkit import Chem
from typing import List, Union
from rdkit.Chem import Descriptors

RDKIT_SMILES_PARSER_PARAMS = Chem.SmilesParserParams()
BondTypes = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]
RDKIT_SMILES_PARSER_PARAMS = Chem.SmilesParserParams()


def str_to_mol(string: str, explicit_hydrogens: bool = True) -> Chem.Mol:
    """
    Converts a SMILES string to an RDKit molecule.

    :param string: The InChI or SMILES string.
    :param explicit_hydrogens: Whether to treat hydrogens explicitly.
    :return: The RDKit molecule.
    """
    if string.startswith('InChI'):
        mol = Chem.MolFromInchi(string, removeHs=not explicit_hydrogens)
    else:
        RDKIT_SMILES_PARSER_PARAMS.removeHs = not explicit_hydrogens
        mol = Chem.MolFromSmiles(string, RDKIT_SMILES_PARSER_PARAMS)

    if explicit_hydrogens:
        return Chem.AddHs(mol)
    else:
        return Chem.RemoveHs(mol)


def CalculateSpinMultiplicity(smiles: str):
    """Calculate spin multiplicity of a molecule. The spin multiplicity is calculated
     from the number of free radical electrons using Hund's rule of maximum
     multiplicity defined as 2S + 1 where S is the total electron spin. The
     total spin is 1/2 the number of free radical electrons in a molecule.

     Arguments:
         smiles: a smiles string

     Returns:
         int : Spin multiplicity.

     """
    mol = Chem.MolFromSmiles(smiles)
    NumRadicalElectrons = 0
    for Atom in mol.GetAtoms():
        NumRadicalElectrons += Atom.GetNumRadicalElectrons()

    TotalElectronicSpin = NumRadicalElectrons / 2
    SpinMultiplicity = 2 * TotalElectronicSpin + 1

    return int(SpinMultiplicity)


def drop_map_num(smi: str):
    """
    Drop the atom map number to get the canonical smiles
    Args:
        smi: the molecule smiles

    Returns:

    """
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def get_decomp_idx(mapped_rsmi, mapped_psmis):
    """
    Get changed atom index of uni-molecule reaction
    A -> B, A -> B + C
    Args:
        mapped_rsmi: Mapped reactant smile
        mapped_psmis: Mapped product smiles

    Returns:

    """
    mapped_pmols = []
    rsmi = drop_map_num(mapped_rsmi)
    rmol = Chem.AddHs(Chem.MolFromSmiles(rsmi))
    mapped_rmol = str_to_mol(mapped_rsmi, explicit_hydrogens=True)
    num_idx_dict = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mapped_rmol.GetAtoms()}
    for psmi in mapped_psmis.split('.'):
        mol = str_to_mol(psmi, explicit_hydrogens=True)
        mapped_pmols.append(mol)

    # Get changed atoms
    breaking, formation = get_changed_bonds([mapped_rsmi], [mapped_psmis])
    bb_idx = [num_idx_dict[num] for num in breaking[0]]
    if formation or len(breaking) > 1:
        return False
    else:
        # Match atoms
        matched_atom_idx = rmol.GetSubstructMatch(mapped_rmol)
        changed_rsmi_idx = [matched_atom_idx[idx] for idx in bb_idx]

        return changed_rsmi_idx


def get_changed_bonds(mapped_rsmis: Union[List[str], str], mapped_psmis: Union[List[str], str]):
    """
    Get broken or formation bonds
    Args:
        mapped_rsmis: reactant SMILES
        mapped_psmis: product SMILES

    Returns: broken and formation bond list
    """
    if type(mapped_rsmis).__name__ == 'list':
        mapped_rmols, mapped_pmols = str_to_mol('.'.join(mapped_rsmis)), str_to_mol('.'.join(mapped_psmis))
    else:
        mapped_rmols, mapped_pmols = str_to_mol(mapped_rsmis), str_to_mol(mapped_psmis)
    r_connect_dict = {atom.GetAtomMapNum(): [a.GetAtomMapNum() for a in atom.GetNeighbors()] \
                      for atom in mapped_rmols.GetAtoms()}
    p_connect_dict = {atom.GetAtomMapNum(): [a.GetAtomMapNum() for a in atom.GetNeighbors()] \
                      for atom in mapped_pmols.GetAtoms()}
    assert sorted(list(r_connect_dict.keys())) == \
           sorted(list(p_connect_dict.keys())), "Unknown map number in reactants or products"
    atom_nums = sorted(list(r_connect_dict.keys()))
    broken = []
    formation = []
    for atom_num in atom_nums:
        for a in r_connect_dict[atom_num]:
            if a not in p_connect_dict[atom_num]:
                bbond = tuple(sorted([atom_num, a]))
                if bbond not in broken:
                    broken.append(bbond)
        for b in p_connect_dict[atom_num]:
            if b not in r_connect_dict[atom_num]:
                fbond = tuple(sorted([atom_num, b]))
                if fbond not in formation:
                    formation.append(fbond)
    return broken, formation


def is_mapped(rxn: str) -> bool:
    """
    Checking atom mapping between reactants and products
    Args:
        rxn: atom mapped reaction smiles

    Returns: bool

    """
    rsmi, psmi = rxn.split('>')[0], rxn.split('>')[-1]
    rmap = [atom.GetAtomMapNum() for atom in str_to_mol(rsmi).GetAtoms()]
    pmap = [atom.GetAtomMapNum() for atom in str_to_mol(psmi).GetAtoms()]
    if len(rmap) != len(set(rmap)) or len(pmap) != len(set(pmap)):
        return False
    if sorted(rmap) != sorted(pmap):
        return False
    else:
        return True


def smarts_to_species(smarts: str) -> List:
    """
    Transform reaction smarts to species
    Args:
        smarts: reaction smarts
    Returns: an unmapped smiles list

    """
    rsmis = smarts.split('>>')[0]
    psmis = smarts.split('>>')[-1]
    smis = []
    for rsmi in rsmis.split('.'):
        smis.append(drop_map_num(rsmi))
    for psmi in psmis.split('.'):
        smis.append(drop_map_num(psmi))

    return smis


def check_rxn(rxn_smiles: str):
    """
    Checking the reaction type
    Args:
        rxn_smiles: the reaction SMILES

    Returns: The reaction type Literal['autodE', 'migration', 'decomposition', 'others']
    """
    # We firstly check the radicals
    rsmi, psmi = rxn_smiles.split('>>')
    rmol, pmol = str_to_mol(rsmi), str_to_mol(psmi)
    if Descriptors.NumRadicalElectrons(rmol) == Descriptors.NumRadicalElectrons(pmol):
        # If the radical numbers of reactants and products are equal
        # the ts can be searched by autodE
        return 'ade'
    # To assign the reaction type, get change bonds firstly
    bbonds, fbonds = get_changed_bonds(mapped_rsmis=rsmi, mapped_psmis=psmi)
    # Get decomposition features. We should get A->B+C type.
    # There might be A -> B type with one H atom migration
    # But we don't consider this in current version.
    # If one bond is broken, this molecule must be decomposed
    if len(bbonds) == 1 and len(fbonds) == 0:
        return 'decomp'
    elif len(bbonds) == 0 and len(fbonds) == 1:
        return 'decomp_rev'

    # Get migration type features. A+B->C+D
    # There might have various types of migration,
    # but we only calculate the hydrogen atom migration type
    if len(bbonds) == 1 and len(fbonds) == 1:
        intersection = set(bbonds[0]) & set(fbonds[0])
        if len(intersection) != 1:  # for H migration, there must be a common H atom
            return 'others'
        else:
            inter_atom_num = intersection.pop()
            for atom in rmol.GetAtoms():
                if atom.GetAtomMapNum() == inter_atom_num:
                    if atom.GetSymbol() == 'H':
                        return 'mig'
                    else:
                        return 'others'
    return 'others'

