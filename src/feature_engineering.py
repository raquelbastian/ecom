import pandas as pd
import os
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Path to saved preprocessor
_encoder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/processed/preprocessor.joblib')
_abs_encoder_path = os.path.abspath(_encoder_path)

# New: PCA-based dimensionality reduction + PCA-backed similarity matrix
# These functions add an alternative similarity artifact that computes cosine
# similarity on PCA-reduced encoded features and does not overwrite the
# existing cosine_sim.npy produced by the original pipeline.

_pca_artifact_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed/pca.joblib'))
_abs_sim_cache_pca = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed/cosine_sim_pca.npy'))


def build_and_save_preprocessor(df: pd.DataFrame, cat_cols: list[str], num_cols: list[str] | None = None, *, save_path: str = _abs_encoder_path, sparse: bool = False, scaler_type: str = 'standard'):
    """
    Fit a ColumnTransformer with OneHotEncoder for categorical columns and a scaler for numeric columns, then save with joblib.
    scaler_type: 'standard' for StandardScaler, 'minmax' for MinMaxScaler
    Returns (preprocessor, encoded_df).
    """
    transformers = []
    remainder = 'passthrough'
    if cat_cols:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=sparse), cat_cols))
    if num_cols is not None and len(num_cols) > 0:
        if scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        transformers.append(('num', scaler, num_cols))
        remainder = 'drop'
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder=remainder
    )
    X = preprocessor.fit_transform(df)
    try:
        out_cols = preprocessor.get_feature_names_out()
    except Exception:
        out_cols = [f'col_{i}' for i in range(X.shape[1])]
    if sparse:
        X = X.toarray()
    df_encoded = pd.DataFrame(X, columns=out_cols, index=df.index)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(preprocessor, save_path)
    return preprocessor, df_encoded

def compute_correlation_matrix(df, columns, plot=True, fill_missing=True):
    """
    Compute and optionally plot the correlation matrix for selected columns.
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of column names to include in the correlation matrix.
        plot (bool): Whether to plot the heatmap. Default True.
        fill_missing (bool): Whether to fill missing values with sensible defaults. Default True.
    Returns:
        pd.DataFrame: The correlation matrix.
    """
    selected_df = df[columns].copy()
    for col in columns:
        selected_df[col] = pd.to_numeric(selected_df[col], errors='coerce')
        if fill_missing:
            if col == 'rating_count':
                selected_df[col] = selected_df[col].fillna(0)
            elif col == 'rating':
                selected_df[col] = selected_df[col].fillna(selected_df[col].median())
            else:
                selected_df[col] = selected_df[col].fillna(selected_df[col].mean())
    correlation_matrix = selected_df.corr()
    if plot:
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Matrix of Numerical Columns')
        plt.show()
    return correlation_matrix

def build_and_save_pca(df_encoded: pd.DataFrame, n_components: int = 50, save_path: str = _pca_artifact_path):
    """Fit PCA on encoded features, persist the fitted PCA, and return (pca, reduced_df).

    - df_encoded: DataFrame of encoded features (dense numeric).
    - n_components: requested number of PCA components (will be clipped to valid range).
    - save_path: where to persist the fitted PCA (joblib).
    """
    # import locally to avoid changing module-level imports
    from sklearn.decomposition import PCA

    if df_encoded is None or len(df_encoded) == 0:
        raise ValueError("df_encoded must be a non-empty DataFrame")

    X = df_encoded.values
    n_samples, n_features = X.shape
    max_components = min(n_samples, n_features)
    k = max(1, min(int(n_components), max_components))

    pca = PCA(n_components=k)
    X_red = pca.fit_transform(X)

    # convert to DataFrame with deterministic column names
    cols = [f'pca_{i}' for i in range(X_red.shape[1])]
    df_red = pd.DataFrame(X_red, columns=cols, index=df_encoded.index)

    # persist PCA
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(pca, save_path)

    return pca, df_red
