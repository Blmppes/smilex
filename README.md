# SMILES Feature Extraction CLI

This command-line tool extracts molecular features from SMILES strings and performs dimensionality reduction using autoencoders (AE, VAE).

## 📦 Requirements

- Python 3.8+
- RDKit
- NumPy, pandas
- scikit-learn
- PyTorch (for VAE)
- TensorFlow (for AE)

You can install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

cd smiles_features

python my_library.py --mode MODE --input INPUT.csv --output OUTPUT [options]


## Modes

### 1. global: Extract RDKit2D descriptors (200 features)

python my_library.py --mode global --input input.csv --output global_features --output_type csv --normalization minmax --fill_nan True

Required:

SMILES column in your input CSV

Optional:

--normalization = minmax, standardscaler, CDF

--fill_nan = True or False

### 2. local: Extract atom/bond features from molecules

python my_library.py --mode local --input input.csv --output local_features --onehot True --pca True

Outputs:

atom_features.csv

bond_features.csv

### 3. vae: Compress global features using Variational Autoencoder

python my_library.py --mode vae --input global_features.csv --output vae_latent --latent_dim 32 --epochs 1000

Output:

vae_latent.npy

### 4. ae: Compress global features using Autoencoder (AE)

python my_library.py --mode ae --input global_features.csv --output ae_latent --latent_dim 16 --epochs 10

Output:

ae_latent.npy

ae_latent_encoder.h5


## Contact

Feel free to fork, contribute, or reach out if you find this useful!

