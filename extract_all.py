import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from global_features import process_global_feature_extraction
from local_features import extract_local_features

def extract_all_features(input_path, output_path, normalization='False', fill_nan=False):
  print("Reading SMILES file...")
  df = pd.read_csv(input_path).dropna(subset=["SMILES"])
  smiles_list = df["SMILES"].tolist()
  ids = df.index.tolist()
  
  print("Extracting local features (onehot=True, pca=True)...")
  local_batch = extract_local_features(smiles_list, onehot=True, pca=True, ids=ids)

  atom_df = pd.DataFrame(local_batch.f_atoms_pca).add_prefix("A_")
  bond_df = pd.DataFrame(local_batch.f_bonds_pca).add_prefix("B_")

#seed(s)
smilesF = pd.read_csv('/kaggle/input/a-lot-of-compounds/SMILES_200Features.csv')

sheader = []

smilesF = smilesF.fillna(0)

#smilesF2 = smilesF.transpose()
sheader = list(smilesF.columns.values)
scaler = preprocessing.StandardScaler()
scaled = scaler.fit_transform(smilesF)
scaled_df = pd.DataFrame(scaled, columns = sheader)
#scaled_df = scaled_df.transpose()

atom_features = atom_features.add_prefix('A_')

bond_features = bond_features.add_prefix('B_')

for i in range (0,12):
  i = str(i)
  column = 'B_'+i
  b1 = bond_features[column]
  
  name = column
  i = int(i)
  atom_features.insert(i,name,b1)

header_smilesf = []
header_smilesf = list(smilesF.columns.values)

#tf.random.set_seed(s)
#seed(s)
second_pca = PCA(n_components = 50) 
data_atom_pca = second_pca.fit_transform(atom_features)

pcaNames = []
for p in range(1,51):
  pc = str(p)
  pca = 'PCA'+pc
  pcaNames.append(pca)


data_atom_pca = pd.DataFrame(data=data_atom_pca, columns=pcaNames)

j = 0
for col in pcaNames:
  col_data = data_atom_pca[col]
  scaled_df.insert(j,col,col_data)
  
  j = j+1


sel = VarianceThreshold(0)
cleaned = sel.fit_transform(scaled_df)
