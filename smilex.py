import argparse
import pandas as pd
from local_features import BatchLocalFeatures
from global_features import process_global_feature_extraction
from vae_encoder import run_vae_encoding
from ae_encoder import run_ae_encoding

def extract_local_features(input_path, output_path, onehot, pca):
    print(f"Reading input file from {input_path}...")
    df = pd.read_csv(input_path).dropna(subset=["SMILES"])

    smiles_list = df["SMILES"].tolist()
    ids = df.index.tolist()  # Or df['compound_stem'] if available

    print(f"Extracting local features (onehot={onehot}, pca={pca})...")
    batch = BatchLocalFeatures(smiles_list, onehot=onehot, pca=pca, ids=ids)

    base = output_path.rsplit('.', 1)[0]  # Remove extension if exists

    if pca:
        pd.DataFrame(batch.f_atoms_pca).to_csv(f"{base}_atom_PCA.csv", index=False)
        pd.DataFrame(batch.f_bonds_pca).to_csv(f"{base}_bond_PCA.csv", index=False)
    else:
        pd.DataFrame(batch.f_atoms).to_csv(f"{base}_atom.csv", index=False)
        pd.DataFrame(batch.f_bonds).to_csv(f"{base}_bond.csv", index=False)

    print("Feature extraction completed and files saved.")


import argparse

def main():
    parser = argparse.ArgumentParser(description="SMILES Feature Extraction Library")
    parser.add_argument('--input', required=True, help='Input CSV file with SMILES column.')
    parser.add_argument('--output', required=True, help='Base output file name (no extension).')
    parser.add_argument('--output_type', choices=['csv', 'parquet'], default='csv', help='Output file format.')
    parser.add_argument('--mode', choices=['local', 'global', 'vae', 'ae'], required=True, help='Feature extraction mode.')

    # Parse initial arguments to check mode
    args, remaining_args = parser.parse_known_args()

    # Add mode-specific arguments
    if args.mode == 'local':
        parser.add_argument('--onehot', type=bool, default=False, help='Use one-hot encoding.')
        parser.add_argument('--pca', type=bool, default=False, help='Apply PCA to local features.')
    elif args.mode == 'global':
        parser.add_argument('--normalization', choices=['False', 'CDF', 'minmax', 'standardscaler', 'robustscaler'],
                            default='False', help='Normalization method for global features.')
        parser.add_argument('--fill_nan', type=bool, default=False, help='Fill NaNs with median.')
    elif args.mode in ['vae', 'ae']:
        parser.add_argument('--latent_dim', type=int, default=32, help='Latent space dimension.')
        parser.add_argument('--epochs', type=int, default=1000 if args.mode == 'vae' else 10,
                            help='Number of training epochs.')

    # Parse full arguments now
    args = parser.parse_args()

    if args.mode == 'local':
        extract_local_features(args.input, args.output, args.onehot, args.pca)
    elif args.mode == 'global':
        process_global_feature_extraction(
            input_path=args.input,
            output_path=args.output,
            output_type=args.output_type,
            normalization=args.normalization,
            fill_nan=args.fill_nan
        )
    elif args.mode == 'vae':
        run_vae_encoding(
            input_csv=args.input,
            output_file=args.output,
            latent_dim=args.latent_dim,
            epochs=args.epochs
        )
    elif args.mode == 'ae':
        run_ae_encoding(
            input_csv=args.input,
            output_file=args.output,
            latent_dim=args.latent_dim,
            epochs=args.epochs
        )

if __name__ == '__main__':
    main()
