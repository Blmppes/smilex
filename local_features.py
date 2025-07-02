# local_features.py

import numpy as np
import pandas as pd
from typing import List, Union
from rdkit import Chem
from sklearn.decomposition import PCA

# --- One-hot encoding ---
def one_hot_encoding(value, choices):
    encoding = [0] * len(choices)
    index = choices.index(value) if value in choices else -1
    if index != -1:
        encoding[index] = 1
    return encoding

# --- Atom feature mappings ---
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

# --- Atom features ---
def atom_features_raw(atom):
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

def atom_features_onehot(atom):
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

# --- Bond features ---
def bond_features_raw(bond):
    bt = bond.GetBondType()
    btt = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3
    }.get(bt, -1)

    return [
        btt,
        bond.GetIsConjugated() if bt is not None else 0,
        bond.IsInRing() if bt is not None else 0,
        int(bond.GetStereo())
    ]

def bond_features_onehot(bond):
    bt = bond.GetBondType()
    return [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated() if bt is not None else 0,
        bond.IsInRing() if bt is not None else 0
    ] + one_hot_encoding(int(bond.GetStereo()), list(range(6)))

# --- Feature class for one molecule ---
class LocalFeatures:
    def __init__(self, mol, onehot=False, pca=False, ids=None):
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)

        self.mol = mol
        self.onehot = onehot

        self.f_atoms = [
            atom_features_onehot(atom) if onehot else atom_features_raw(atom)
            for atom in mol.GetAtoms()
        ]
        self.f_bonds = [
            bond_features_onehot(bond) if onehot else bond_features_raw(bond)
            for bond in mol.GetBonds()
        ]

        self.n_atoms = len(self.f_atoms)
        self.n_bonds = len(self.f_bonds)
        self.f_atoms_dim = len(self.f_atoms[0]) if self.f_atoms else 0
        self.f_bonds_dim = len(self.f_bonds[0]) if self.f_bonds else 0

        self.f_atoms_pca = []
        self.f_bonds_pca = []

        if pca and self.f_atoms and self.f_bonds:
            self.f_atoms_pca = PCA(n_components=1).fit_transform(np.array(self.f_atoms).T).T
            self.f_bonds_pca = PCA(n_components=1).fit_transform(np.array(self.f_bonds).T).T

        self.mol_id_atoms = [ids] * self.n_atoms if ids is not None else []
        self.mol_id_bonds = [ids] * self.n_bonds if ids is not None else []

# --- Feature class for batch of molecules ---
class BatchLocalFeatures:
    def __init__(self, smiles_list, onehot=False, pca=False, ids=None):
        self.mol_graphs = [LocalFeatures(sm, onehot, pca, ids[i] if ids else None)
                           for i, sm in enumerate(smiles_list)]

        self.f_atoms = sum((g.f_atoms for g in self.mol_graphs), [])
        self.f_bonds = sum((g.f_bonds for g in self.mol_graphs), [])
        self.f_atoms_pca = sum((g.f_atoms_pca.tolist() for g in self.mol_graphs if g.f_atoms_pca != []), [])
        self.f_bonds_pca = sum((g.f_bonds_pca.tolist() for g in self.mol_graphs if g.f_bonds_pca != []), [])
        self.f_atoms_id = sum((g.mol_id_atoms for g in self.mol_graphs), [])
        self.f_bonds_id = sum((g.mol_id_bonds for g in self.mol_graphs), [])

        self.n_atoms = len(self.f_atoms)
        self.n_bonds = len(self.f_bonds)
        self.a_scope = [(sum(g.n_atoms for g in self.mol_graphs[:i]), g.n_atoms)
                        for i, g in enumerate(self.mol_graphs)]
        self.b_scope = [(sum(g.n_bonds for g in self.mol_graphs[:i]), g.n_bonds)
                        for i, g in enumerate(self.mol_graphs)]

# --- Entry function ---
def mol2local(mols, onehot=False, pca=False, ids=None):
    return BatchLocalFeatures(mols, onehot=onehot, pca=pca, ids=ids)
