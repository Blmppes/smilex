import pandas as pd
import numpy as np
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


# ===================== Descriptor Generator =====================

generator = MakeGenerator(("RDKit2D",))


def extract_global_features(smiles: str):
    """Extract 200 RDKit 2D descriptors."""
    try:
        data = generator.process(smiles)
        if data[0]:  # SMILES valid
            return data[1:]
    except Exception:
        pass
    return None


def handle_missing_values(df: pd.DataFrame, fill_nan: bool) -> pd.DataFrame:
    if df.isnull().any().any() or np.isinf(df).any().any():
        print("There are missing or infinite values in the data. Handling...")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        if fill_nan:
            df = df.fillna(df.median(numeric_only=True))
    return df


def normalize_features(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == 'CDF':
        return df.rank(method='average', pct=True)
    elif method == 'minmax':
        return pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)
    elif method == 'standardscaler':
        return pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)
    elif method == 'robustscaler':
        return pd.DataFrame(RobustScaler().fit_transform(df), columns=df.columns)
    return df


def process_global_feature_extraction(input_path: str, output_path: str,
                                      output_type: str = 'csv',
                                      normalization: str = 'False',
                                      fill_nan: bool = False):
    print(f"Loading SMILES from {input_path}...")
    df = pd.read_csv(input_path)
    if 'SMILES' not in df.columns:
        raise ValueError("Missing 'SMILES' column in input CSV.")
    if df['SMILES'].isnull().any():
        raise ValueError("Null values found in 'SMILES' column.")

    print("Extracting global features...")
    features = df['SMILES'].apply(extract_global_features)
    column_names = [name for name, _ in generator.GetColumns()[1:]]
    features_df = pd.DataFrame(features.tolist(), columns=column_names)

    features_df = handle_missing_values(features_df, fill_nan)

    if normalization and normalization != 'False':
        print(f"Applying {normalization} normalization...")
        features_df = normalize_features(features_df, normalization)

    print(f"Saving features to {output_path}.{output_type}...")
    if output_type == 'csv':
        features_df.to_csv(f"{output_path}.csv", index=False)
    elif output_type == 'parquet':
        features_df.to_parquet(f"{output_path}.parquet", index=False)
    else:
        raise ValueError("Unsupported output type. Choose 'csv' or 'parquet'.")
    
    print("Done.")
    return features_df
