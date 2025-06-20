import numpy as np
from typing import List, Union
from rdkit import Chem
from sklearn.decomposition import PCA


# ===================== Atom and Bond Feature Definitions =====================

ATOM_FEATURES = {
    'atomic_num': list(range(118)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}


def one_hot_encoding(value, choices):
    encoding = [0] * len(choices)
    index = choices.index(value) if value in choices else -1
    if index != -1:
        encoding[index] = 1
    return encoding


# ===================== Atom and Bond Feature Extraction =====================

def atom_features_raw(atom: Chem.rdchem.Atom) -> List[Union[int, float]]:
    return [
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        atom.GetFormalCharge(),
        int(atom.GetChiralTag()),
        int(atom.GetTotalNumHs()),
        int(atom.GetHybridization()),
        atom.GetIsAromatic(),
        atom.GetMass()
    ]


def atom_features_onehot(atom: Chem.rdchem.Atom) -> List[Union[int, float]]:
    return (
        one_hot_encoding(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) +
        one_hot_encoding(atom.GetTotalDegree(), ATOM_FEATURES['degree']) +
        one_hot_encoding(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) +
        one_hot_encoding(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) +
        one_hot_encoding(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) +
        one_hot_encoding(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) +
        [1 if atom.GetIsAromatic() else 0] +
        [atom.GetMass() * 0.01]
    )


def bond_features_raw(bond: Chem.rdchem.Bond) -> List[Union[int, float]]:
    bt = bond.GetBondType()
    btt = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3
    }.get(bt, -1)

    return [
        btt,
        int(bond.GetIsConjugated()) if bt else 0,
        int(bond.IsInRing()) if bt else 0,
        int(bond.GetStereo())
    ]


def bond_features_onehot(bond: Chem.rdchem.Bond) -> List[int]:
    bt = bond.GetBondType()
    return [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        int(bond.GetIsConjugated()) if bt else 0,
        int(bond.IsInRing()) if bt else 0,
        *one_hot_encoding(int(bond.GetStereo()), list(range(6)))
    ]


# ===================== Feature Extraction Classes =====================

class LocalFeatures:
    def __init__(self, mol: Union[str, Chem.Mol], onehot=False, pca=False, mol_id=None):
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        self.mol = mol
        self.mol_id = mol_id
        self.onehot = onehot
        self.pca_enabled = pca

        # Extract features
        self.f_atoms = self._extract_atom_features()
        self.f_bonds = self._extract_bond_features()

        self.n_atoms = len(self.f_atoms)
        self.n_bonds = len(self.f_bonds)

        self.f_atoms_pca = self._apply_pca(self.f_atoms) if pca else []
        self.f_bonds_pca = self._apply_pca(self.f_bonds) if pca else []

        self.mol_id_atoms = [mol_id] * self.n_atoms if mol_id else []
        self.mol_id_bonds = [mol_id] * self.n_bonds if mol_id else []

    def _extract_atom_features(self):
        extractor = atom_features_onehot if self.onehot else atom_features_raw
        return [extractor(atom) for atom in self.mol.GetAtoms()]

    def _extract_bond_features(self):
        extractor = bond_features_onehot if self.onehot else bond_features_raw
        return [extractor(bond) for bond in self.mol.GetBonds()]

    def _apply_pca(self, features):
        pca_model = PCA(n_components=1)
        return pca_model.fit_transform(np.array(features)).flatten().tolist()


class BatchLocalFeatures:
    def __init__(self, mols: List[Union[str, Chem.Mol]], onehot=False, pca=False, ids=None):
        self.mol_graphs = [
            LocalFeatures(mol, onehot, pca, mol_id=(ids[i] if ids else None))
            for i, mol in enumerate(mols)
        ]
        self._aggregate_features()

    def _aggregate_features(self):
        self.f_atoms = []
        self.f_bonds = []
        self.f_atoms_pca = []
        self.f_bonds_pca = []
        self.f_atoms_id = []
        self.f_bonds_id = []
        self.a_scope = []
        self.b_scope = []
        self.n_atoms = 0
        self.n_bonds = 0

        for graph in self.mol_graphs:
            self.f_atoms.extend(graph.f_atoms)
            self.f_bonds.extend(graph.f_bonds)
            self.f_atoms_pca.extend(graph.f_atoms_pca)
            self.f_bonds_pca.extend(graph.f_bonds_pca)
            self.f_atoms_id.extend(graph.mol_id_atoms)
            self.f_bonds_id.extend(graph.mol_id_bonds)
            self.a_scope.append((self.n_atoms, graph.n_atoms))
            self.b_scope.append((self.n_bonds, graph.n_bonds))
            self.n_atoms += graph.n_atoms
            self.n_bonds += graph.n_bonds
