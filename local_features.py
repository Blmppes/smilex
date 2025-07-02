import numpy as np
import pandas as pd
import sklearn
import rdkit
from typing import List, Tuple, Union
from rdkit import Chem
from sklearn.decomposition import PCA

def one_hot_encoding(value, choices):
    encoding = [0] * (len(choices))
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding

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
    ]
}

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
    return one_hot_encoding(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
           one_hot_encoding(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
           one_hot_encoding(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
           one_hot_encoding(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
           one_hot_encoding(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
           one_hot_encoding(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01]

def bond_features_raw(bond):
    bt = bond.GetBondType()
    if bt == Chem.rdchem.BondType.SINGLE: btt = 0
    elif bt == Chem.rdchem.BondType.DOUBLE: btt = 1
    elif bt == Chem.rdchem.BondType.TRIPLE: btt = 2
    elif bt == Chem.rdchem.BondType.AROMATIC: btt = 3
    else: btt = -1
    return [
        btt,
        bond.GetIsConjugated() if bt is not None else 0,
        bond.IsInRing() if bt is not None else 0,
        int(bond.GetStereo())
    ]

def bond_features_onehot(bond):
    bt = bond.GetBondType()
    fbond = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated() if bt is not None else 0,
        bond.IsInRing() if bt is not None else 0
    ]
    fbond += one_hot_encoding(int(bond.GetStereo()), list(range(6)))
    return fbond

class LocalFeatures:
    def __init__(self, mol, onehot=False, pca=False, ids=None):
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        self.mol = mol
        self.onehot = onehot
        self.n_atoms = 0
        self.n_bonds = 0
        self.f_atoms = []
        self.f_bonds = []
        self.f_atoms_pca = []
        self.f_bonds_pca = []
        self.mol_id_atoms = []
        self.mol_id_bonds = []

        if onehot:
            self.f_atoms = [atom_features_onehot(atom) for atom in mol.GetAtoms()]
            self.f_bonds = [bond_features_onehot(bond) for bond in mol.GetBonds()]
        else:
            self.f_atoms = [atom_features_raw(atom) for atom in mol.GetAtoms()]
            self.f_bonds = [bond_features_raw(bond) for bond in mol.GetBonds()]

        self.n_atoms = len(self.f_atoms)
        self.n_bonds = len(self.f_bonds)
        self.f_atoms_dim = np.shape(self.f_atoms)[1] if self.n_atoms > 0 else 0
        self.f_bonds_dim = np.shape(self.f_bonds)[1] if self.n_bonds > 0 else 0

        if pca and self.n_atoms > 0 and self.n_bonds > 0:
            fa = np.array(self.f_atoms).T
            fb = np.array(self.f_bonds).T
            pca_model = PCA(n_components=1)
            pc_atoms = pca_model.fit_transform(fa)
            pc_bonds = pca_model.fit_transform(fb)
            self.f_atoms_pca = pc_atoms.T.tolist()
            self.f_bonds_pca = pc_bonds.T.tolist()

        if ids is not None:
            self.mol_id_atoms = [ids] * self.n_atoms
            self.mol_id_bonds = [ids] * self.n_bonds

class BatchLocalFeatures:
    def __init__(self, mol_graphs):
        self.mol_graphs = mol_graphs
        self.n_atoms = 0
        self.n_bonds = 0
        self.a_scope = []
        self.b_scope = []
        self.f_atoms = []
        self.f_bonds = []
        self.f_atoms_pca = []
        self.f_bonds_pca = []
        self.f_atoms_id = []
        self.f_bonds_id = []

        for g in mol_graphs:
            self.f_atoms.extend(g.f_atoms)
            self.f_bonds.extend(g.f_bonds)
            self.f_atoms_pca.extend(g.f_atoms_pca)
            self.f_bonds_pca.extend(g.f_bonds_pca)
            self.f_atoms_id.extend(g.mol_id_atoms)
            self.f_bonds_id.extend(g.mol_id_bonds)
            self.a_scope.append((self.n_atoms, g.n_atoms))
            self.b_scope.append((self.n_bonds, g.n_bonds))
            self.n_atoms += g.n_atoms
            self.n_bonds += g.n_bonds

def mol2local(mols, onehot=False, pca=False, ids=None):
    if ids is not None:
        return BatchLocalFeatures([LocalFeatures(mol, onehot, pca, iid) for mol, iid in zip(mols, ids)])
    else:
        return BatchLocalFeatures([LocalFeatures(mol, onehot, pca) for mol in mols])

def extract_local_features(input_path, output_path, onehot=False, pca=False):
    print(f"Reading input file from {input_path}...")
    df = pd.read_csv(input_path).dropna(subset=["SMILES"])
    smiles_list = df["SMILES"].tolist()
    ids = df.index.tolist()
    print(f"Extracting local features (onehot={onehot}, pca={pca})...")
    batch = mol2local(smiles_list, onehot=onehot, pca=pca, ids=ids)
    base = output_path.rsplit('.', 1)[0]
    if pca:
        pd.DataFrame(batch.f_atoms_pca).to_csv(f"{base}_atom_PCA.csv", index=False)
        pd.DataFrame(batch.f_bonds_pca).to_csv(f"{base}_bond_PCA.csv", index=False)
    else:
        pd.DataFrame(batch.f_atoms).to_csv(f"{base}_atom.csv", index=False)
        pd.DataFrame(batch.f_bonds).to_csv(f"{base}_bond.csv", index=False)
    print("Feature extraction completed and files saved.")
