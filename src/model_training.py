import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

# Third-party imports: attempt to import but make module import-safe if missing
SKLEARN_AVAILABLE = True
try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import NearestNeighbors
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import PCA, LatentDirichletAllocation
    from scipy.sparse.linalg import svds
except Exception as _e:
    SKLEARN_AVAILABLE = False
    # Defer raising until runtime so importing this module won't crash the notebook
    def _missing_sklearn(*args, **kwargs):
        raise RuntimeError("scikit-learn (or one of its submodules) is required for this function. "
                           "Install scikit-learn in your environment to use model_training features.")
    cosine_similarity = _missing_sklearn
    OneHotEncoder = None
    ColumnTransformer = None
    LinearRegression = None
    NearestNeighbors = None
    TfidfVectorizer = None
    CountVectorizer = None
    PCA = None
    LatentDirichletAllocation = None
    svds = None

# Optional libraries
try:
    import joblib
except Exception:
    joblib = None

try:
    from joblib import Parallel, delayed
except Exception:
    Parallel = None
    delayed = None

try:
    from textblob import TextBlob
except Exception:
    TextBlob = None


# Global DataFrame, to be populated by the calling script/notebook
df = None

# Path to cached similarity matrix file
_sim_cache_path = os.path.join(os.path.dirname(__file__), '../app/dataset/cosine_sim.npy')
_abs_sim_cache_path = os.path.abspath(_sim_cache_path)

# Path to saved preprocessor
_encoder_path = os.path.join(os.path.dirname(__file__), '../app/dataset/preprocessor.joblib')
_abs_encoder_path = os.path.abspath(_encoder_path)

# Additional artifact paths (parquet + index map)
_product_features_parquet = os.path.abspath(os.path.join(os.path.dirname(__file__), '../app/dataset/product_features.parquet'))
_encoded_features_parquet = os.path.abspath(os.path.join(os.path.dirname(__file__), '../app/dataset/encoded_features.parquet'))
_index_map_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../app/dataset/index_map.json'))

# In-memory caches
_df_product_features_cache = None
_encoded_features_cache = None
_index_map_cache = None

def save_similarity_to_file(matrix: np.ndarray, path: str = _abs_sim_cache_path) -> None:
    """Save similarity matrix to disk as a .npy file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, matrix)

def load_similarity_from_file(path: str = _abs_sim_cache_path) -> np.ndarray:
    """Load a similarity matrix saved with np.save/.npy. Use mmap to avoid large memory spikes."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Similarity file not found: {path}")
    # use mmap_mode='r' for large matrices so we don't load entire file into memory
    return np.load(path, mmap_mode='r')

def load_or_build_similarity(path: str = _abs_sim_cache_path, feature_df: pd.DataFrame = None) -> np.ndarray:
    """Try to load the similarity matrix from `path`. If not present, compute from `feature_df`, save it, and return it."""
    try:
        return load_similarity_from_file(path)
    except FileNotFoundError:
        if feature_df is None:
            raise ValueError("feature_df must be provided to build the similarity matrix when cache is missing")
        # ensure we pass a dense numeric array to cosine_similarity
        mat = cosine_similarity(feature_df.values if hasattr(feature_df, 'values') else feature_df)
        save_similarity_to_file(mat, path)
        return mat

def build_and_save_preprocessor(df: pd.DataFrame, cat_cols: list[str], num_cols: list[str] | None = None, *, save_path: str = _abs_encoder_path, sparse: bool = False):
    """Fit a ColumnTransformer with OneHotEncoder on `df` and save it with joblib.

    Returns (preprocessor, encoded_df).
    - df: original DataFrame used to fit the transformer.
    - cat_cols: list of categorical column names to encode.
    - num_cols: list of numeric columns to keep (if None, remainder='passthrough' keeps all others).
    - save_path: where to persist the fitted preprocessor.
    - sparse: if True, OneHotEncoder returns a sparse matrix (memory efficient for many categories).
    """
    # Build transformer list and remainder behavior. If num_cols is provided, include
    # an explicit passthrough transformer for them and drop the remainder. Otherwise
    # passthrough all other (non-categorical) columns.
    transformers = []
    remainder = 'passthrough'
    if cat_cols:
        transformers.append(('cat', None, cat_cols))

    if num_cols is not None and len(num_cols) > 0:
        # Explicitly passthrough numeric columns and drop any other remainder.
        transformers.append(('num', 'passthrough', num_cols))
        remainder = 'drop'

    # Create OneHotEncoder with compatibility for different sklearn versions
    try:
        # older sklearn versions accept `sparse`
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=sparse)
    except TypeError:
        # newer sklearn versions use `sparse_output`
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=sparse)

    # Replace the placeholder cat transformer with the actual OneHotEncoder instance
    # so that the transformer tuple is ('cat', ohe, cat_cols).
    for i, t in enumerate(transformers):
        if t[0] == 'cat':
            transformers[i] = ('cat', ohe, cat_cols)

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder=remainder
    )

    # fit and transform
    X = preprocessor.fit_transform(df)

    # get output column names (sklearn >=1.0)
    try:
        out_cols = preprocessor.get_feature_names_out()
    except Exception:
        # fallback: create generic column names
        out_cols = [f'col_{i}' for i in range(X.shape[1])]

    # convert to DataFrame (if sparse, convert to array first)
    if sparse:
        X = X.toarray()

    df_encoded = pd.DataFrame(X, columns=out_cols, index=df.index)

    # save the fitted preprocessor
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(preprocessor, save_path)

    return preprocessor, df_encoded


def load_preprocessor(path: str = _abs_encoder_path):
    """Load a saved preprocessor (joblib file).

    If loading fails due to scikit-learn cross-version pickling issues (for
    example: "Can't get attribute '_RemainderColsList'"), attempt to rebuild
    the preprocessor from the product features artifact and persist it.
    """
    # If the persisted file is missing, signal the caller to rebuild
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessor file not found: {path}")

    try:
        pre = joblib.load(path)
        return pre
    except AttributeError as e:
        # Common when sklearn internal classes changed between versions
        print(f"Warning: Could not deserialize preprocessor (AttributeError): {e}\n" \
              "Attempting to rebuild the preprocessor from product features...")
    except Exception as e:
        # Generic fallback: try to rebuild as well
        print(f"Warning: Could not load preprocessor: {e}\nAttempting to rebuild the preprocessor from product features...")

    # Attempt to rebuild the preprocessor using the product features parquet
    try:
        df_pf = load_product_features()
    except Exception as e:
        raise RuntimeError(f"Failed to load product features to rebuild preprocessor: {e}")

    try:
        cat_cols = infer_categorical_columns(df_pf)
        pre, encoded = build_and_save_preprocessor(df_pf, cat_cols, num_cols=None, save_path=path, sparse=False)
        print(f"Rebuilt preprocessor and saved to: {path}")
        return pre
    except Exception as e:
        raise RuntimeError(f"Failed to rebuild and save preprocessor: {e}")

def apply_preprocessor(df_new: pd.DataFrame, path: str = _abs_encoder_path):
    """Apply a saved preprocessor to new DataFrame and return an encoded DataFrame.

    Raises FileNotFoundError if preprocessor file is missing.
    """
    pre = load_preprocessor(path)
    X_new = pre.transform(df_new)
    try:
        cols = pre.get_feature_names_out()
    except Exception:
        cols = [f'col_{i}' for i in range(X_new.shape[1])]

    # if result is sparse, convert to dense
    if hasattr(X_new, 'toarray'):
        X_new = X_new.toarray()

    return pd.DataFrame(X_new, columns=cols, index=df_new.index)

def infer_categorical_columns(df: pd.DataFrame) -> list:
    """Return a list of categorical column names inferred from dtypes."""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

# add lazy loader for product_features with parquet fallback
def load_product_features() -> pd.DataFrame:
    """Load product features from the precomputed Parquet artifact only. Cached in-memory.

    This intentionally does NOT fall back to CSV to avoid expensive CSV parsing at runtime.
    If the Parquet artifact is missing, instruct the user to run the build_artifacts script.
    """
    global _df_product_features_cache
    if (_df_product_features_cache is not None):
        return _df_product_features_cache

    # Only load from Parquet artifact; avoid reading the raw CSV at runtime
    if os.path.exists(_product_features_parquet):
        df_pf = pd.read_parquet(_product_features_parquet)
        _df_product_features_cache = df_pf
        return df_pf

    # Parquet artifact missing — do not attempt CSV fallback here
    raise FileNotFoundError(
        f"Encoded product features not found at {_product_features_parquet}.\n"
        "Please run the artifact builder to create the required files:\n"
        "  cd python-services && python3 build_artifacts.py"
    )

def save_encoded_features(df_encoded: pd.DataFrame, path: str = _encoded_features_parquet) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_encoded.to_parquet(path, index=False)

def load_encoded_features(path: str = _encoded_features_parquet) -> pd.DataFrame:
    global _encoded_features_cache
    if _encoded_features_cache is not None:
        return _encoded_features_cache
    if not os.path.exists(path):
        raise FileNotFoundError(f"Encoded features not found: {path}")
    df_enc = pd.read_parquet(path)
    _encoded_features_cache = df_enc
    return df_enc

def save_index_map(index_map: dict, path: str = _index_map_path) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(index_map, f)

def load_index_map(path: str = _index_map_path) -> dict:
    global _index_map_cache
    if _index_map_cache is not None:
        return _index_map_cache
    if not os.path.exists(path):
        raise FileNotFoundError(f"Index map not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        idx_map = json.load(f)
    _index_map_cache = idx_map
    return idx_map

# modify get_feature_matrix to use encoded parquet or build and persist artifacts
def get_feature_matrix():
    """Return an encoded feature DataFrame for similarity computation.

    Preference order:
    1) Load encoded features parquet if available.
    2) If preprocessor exists, apply it to product_features and persist encoded parquet + index map.
    3) Otherwise build preprocessor, persist, and encoded parquet + index map.
    """
    # 1) try encoded parquet
    try:
        return load_encoded_features()
    except Exception:
        pass

    # ensure product features DataFrame is available
    df_pf = load_product_features()

    # 2) try apply saved preprocessor
    if os.path.exists(_abs_encoder_path):
        try:
            encoded = apply_preprocessor(df_pf, _abs_encoder_path)
            # persist encoded and index map
            save_encoded_features(encoded)
            index_map = {str(pid): int(i) for i, pid in enumerate(df_pf['product_id'].astype(str).tolist())}
            save_index_map(index_map)
            return encoded
        except Exception:
            pass

    # 3) build preprocessor and encoded features
    cat_cols = infer_categorical_columns(df_pf)
    pre, encoded = build_and_save_preprocessor(df_pf, cat_cols, num_cols=None, save_path=_abs_encoder_path, sparse=False)
    try:
        save_encoded_features(encoded)
    except Exception:
        pass
    try:
        index_map = {str(pid): int(i) for i, pid in enumerate(df_pf['product_id'].astype(str).tolist())}
        save_index_map(index_map)
    except Exception:
        pass
    return encoded

# adjust build_similarity_matrix to use memmap-friendly loading
def build_similarity_matrix():
    global cosine_sim_matrix
    if (cosine_sim_matrix is None):
        feature_df = get_feature_matrix()
        cosine_sim_matrix = load_or_build_similarity(_abs_sim_cache_path, feature_df)
    return cosine_sim_matrix

def print_head():
    print(df.head())

# Example: Load data (you can adjust the path as needed)
def load_data(csv_path):
    return pd.read_csv(csv_path)

# Example: Train a simple linear regression model
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Example: Make predictions
def make_prediction(model, X):
    return model.predict(X)

cosine_sim_matrix = None


def build_similarity_matrix():
    global cosine_sim_matrix
    if (cosine_sim_matrix is None):
        feature_df = get_feature_matrix()
        cosine_sim_matrix = load_or_build_similarity(_abs_sim_cache_path, feature_df)
    return cosine_sim_matrix

def print_head():
    print(df.head())

# Example: Load data (you can adjust the path as needed)
def load_data(csv_path):
    return pd.read_csv(csv_path)

# Example: Train a simple linear regression model
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Example: Make predictions
def make_prediction(model, X):
    return model.predict(X)


print("Recommendation function 'get_recommendations' updated to use reduced feature set.")

# New: PCA-based dimensionality reduction + PCA-backed similarity matrix
# These functions add an alternative similarity artifact that computes cosine
# similarity on PCA-reduced encoded features and does not overwrite the
# existing cosine_sim.npy produced by the original pipeline.

_pca_artifact_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed/pca.joblib'))
_abs_sim_cache_pca = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed/cosine_sim_pca.npy'))


def build_and_save_pca(df_encoded: pd.DataFrame, n_components: int = 50, save_path: str = _pca_artifact_path):
    """Fit PCA on encoded features, persist the fitted PCA, and return (pca, reduced_df).

    - df_encoded: DataFrame of encoded features (dense numeric).
    - n_components: requested number of PCA components (will be clipped to valid range).
    - save_path: where to persist the fitted PCA (joblib).
    """

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


def load_pca(path: str = _pca_artifact_path):
    """Load persisted PCA object (joblib)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"PCA artifact not found: {path}")
    return joblib.load(path)


def apply_pca_to_encoded(df_encoded: pd.DataFrame, pca_path: str = _pca_artifact_path):
    """Apply a saved PCA to an encoded DataFrame and return a reduced DataFrame."""
    pca = load_pca(pca_path)
    X_red = pca.transform(df_encoded.values)
    cols = [f'pca_{i}' for i in range(X_red.shape[1])]
    return pd.DataFrame(X_red, columns=cols, index=df_encoded.index)


def get_reduced_feature_matrix(n_components: int = 50):
    """Return a PCA-reduced feature DataFrame.

    Preference order:
      1) If a PCA artifact exists, apply it to the encoded features and return reduced frame.
      2) Otherwise build PCA from encoded features, persist it, and return reduced frame.

    This function uses the encoded features produced by get_feature_matrix().
    """
    # 1) ensure encoded features exist
    encoded = None
    try:
        encoded = get_feature_matrix()
    except Exception as e:
        raise RuntimeError(f"Unable to load encoded features for PCA: {e}")

    # If a persisted PCA exists, use it if the dimensionality is compatible
    if os.path.exists(_pca_artifact_path):
        try:
            df_red = apply_pca_to_encoded(encoded, _pca_artifact_path)
            # if the existing PCA has fewer components than requested, that's fine
            return df_red
        except Exception:
            # fall through to rebuild
            pass

    # Build and save PCA
    pca, df_red = build_and_save_pca(encoded, n_components=n_components, save_path=_pca_artifact_path)
    return df_red


def build_similarity_matrix_with_pca(n_components: int = 50):
    """Build or load a cosine similarity matrix computed on PCA-reduced features.

    This stores/loads from `cosine_sim_pca.npy` to avoid clobbering the original matrix.
    """
    # Try loading cached PCA similarity file
    try:
        return load_similarity_from_file(_abs_sim_cache_pca)
    except FileNotFoundError:
        # build reduced feature matrix and compute similarity
        df_red = get_reduced_feature_matrix(n_components=n_components)
        mat = cosine_similarity(df_red.values)
        # persist
        save_similarity_to_file(mat, _abs_sim_cache_pca)
        return mat

# end of PCA additions

# --- Review-based Similarity (TF-IDF on aggregated reviews) ---

_reviews_sim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../app/dataset/cosine_sim_reviews.npy'))
_reviews_index_map_path = _index_map_path  # reuse index map

_reviews_agg_cache = None
_reviews_sim_cache = None

def aggregate_reviews(df_input: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all reviews for each product into a single string."""
    if 'review_content' not in df_input.columns:
        raise ValueError("review_content column not found in DataFrame")
    # Some products have comma-separated reviews in a single row
    def join_reviews(row):
        if pd.isna(row['review_content']):
            return ''
        # If already a list, join; else, split by comma and join
        if isinstance(row['review_content'], list):
            return ' '.join(row['review_content'])
        return ' '.join(str(row['review_content']).split(','))
    agg = df_input[['product_id', 'review_content']].copy()
    agg['all_reviews'] = agg.apply(join_reviews, axis=1)
    return agg[['product_id', 'all_reviews']]

def build_and_save_review_similarity(df_input: pd.DataFrame, path: str = _reviews_sim_path):
    """Compute and save cosine similarity matrix based on aggregated reviews."""
    agg = aggregate_reviews(df_input)
    tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
    X = tfidf.fit_transform(agg['all_reviews'])
    sim = cosine_similarity(X)
    save_similarity_to_file(sim, path)
    # Save index map (product_id -> row index)
    index_map = {str(pid): int(i) for i, pid in enumerate(agg['product_id'].astype(str).tolist())}
    save_index_map(index_map, _reviews_index_map_path)
    return sim

def load_review_similarity(df, path: str = _reviews_sim_path):
    if not os.path.exists(path):
        # Build if missing
        return build_and_save_review_similarity(df, path)
    return load_similarity_from_file(path)

def get_review_recommendations(product_id, N=5):
    """Return top-N recommendations using review-based similarity."""
    sim = load_review_similarity(df, _reviews_sim_path)
    try:
        idx_map = load_index_map(_reviews_index_map_path)
        idx = idx_map.get(str(product_id))
        if idx is None:
            print(f"Product ID '{product_id}' not found in review index map.")
            return []
    except Exception:
        try:
            idx = int(df[df['product_id'] == product_id].index[0])
        except Exception:
            print(f"Product ID '{product_id}' not found.")
            return []
    sim_row = np.array(sim[idx])
    sim_scores = list(enumerate(sim_row))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:N+1]
    product_indices = [i[0] for i in sim_scores]
    # Use the same df order as aggregation
    agg = aggregate_reviews(df)
    rec_ids = agg.iloc[product_indices]['product_id'].tolist()
    # Map back to main df for details
    recommended_products = df[df['product_id'].isin(rec_ids)][['product_id', 'product_name', 'category', 'rating', 'discounted_price', 'img_link']]
    return recommended_products

# --- Content-based Similarity (using all product text fields) ---
def aggregate_product_content(df_input: pd.DataFrame) -> pd.DataFrame:
    """Aggregate product_name, category, about_product, review_title, review_content into a single string per product."""
    def join_fields(row):
        fields = [
            str(row.get('product_name', '')),
            str(row.get('category', '')),
            str(row.get('about_product', '')),
            str(row.get('review_title', '')),
            str(row.get('review_content', '')),
        ]
        # Some fields may have comma-separated lists (e.g., review_title, review_content)
        # Join all comma-separated items with space
        fields = [' '.join(f.split(',')) for f in fields]
        return ' '.join(fields)
    agg = df_input[['product_id', 'product_name', 'category', 'about_product', 'review_title', 'review_content']].copy()
    agg['all_content'] = agg.apply(join_fields, axis=1)
    return agg[['product_id', 'all_content']]

_content_sim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../app/dataset/cosine_sim_content.npy'))

def build_and_save_content_similarity(df_input: pd.DataFrame, path: str = _content_sim_path):
    """Compute and save cosine similarity matrix based on all product content."""
    agg = aggregate_product_content(df_input)
    tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
    X = tfidf.fit_transform(agg['all_content'])
    sim = cosine_similarity(X)
    save_similarity_to_file(sim, path)
    # Save index map (product_id -> row index)
    index_map = {str(pid): int(i) for i, pid in enumerate(agg['product_id'].astype(str).tolist())}
    save_index_map(index_map, _reviews_index_map_path)  # reuse index map path
    return sim

def load_content_similarity(df, path: str = _content_sim_path):
    if not os.path.exists(path):
        return build_and_save_content_similarity(df, path)
    return load_similarity_from_file(path)

def get_content_recommendations(product_id, N=5):
    """Return top-N recommendations using content-based similarity (all text fields)."""
    sim = load_content_similarity(df, _content_sim_path)
    try:
        idx_map = load_index_map(_reviews_index_map_path)
        idx = idx_map.get(str(product_id))
        if idx is None:
            print(f"Product ID '{product_id}' not found in content index map.")
            return []
    except Exception:
        try:
            idx = int(df[df['product_id'] == product_id].index[0])
        except Exception:
            print(f"Product ID '{product_id}' not found.")
            return []
    sim_row = np.array(sim[idx])
    sim_scores = list(enumerate(sim_row))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:N+1]
    product_indices = [i[0] for i in sim_scores]
    agg = aggregate_product_content(df)
    rec_ids = agg.iloc[product_indices]['product_id'].tolist()
    recommended_products = df[df['product_id'].isin(rec_ids)][['product_id', 'product_name', 'category', 'rating', 'discounted_price', 'img_link']]
    return recommended_products

# --- Content-based Similarity with PCA (TF-IDF + PCA) ---

_content_pca_sim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../app/dataset/cosine_sim_content_pca.npy'))


def build_and_save_content_pca_similarity(df_input: pd.DataFrame, path: str = _content_pca_sim_path, n_components: int = 50):
    """Compute and save cosine similarity matrix based on PCA-reduced TF-IDF of all product content."""
    agg = aggregate_product_content(df_input)
    tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
    X = tfidf.fit_transform(agg['all_content'])
    # Convert to dense array for PCA
    X_dense = X.toarray()
    n_samples, n_features = X_dense.shape
    k = max(1, min(int(n_components), min(n_samples, n_features)))
    pca = PCA(n_components=k)
    X_pca = pca.fit_transform(X_dense)
    sim = cosine_similarity(X_pca)
    save_similarity_to_file(sim, path)
    # Save index map (product_id -> row index)
    index_map = {str(pid): int(i) for i, pid in enumerate(agg['product_id'].astype(str).tolist())}
    save_index_map(index_map, _reviews_index_map_path)
    return sim

def load_content_pca_similarity(df, path: str = _content_pca_sim_path, n_components: int = 50):
    if not os.path.exists(path):
        return build_and_save_content_pca_similarity(df, path, n_components=n_components)
    return load_similarity_from_file(path)

def get_content_recommendations_pca(product_id, N=5, n_components: int = 50):
    """Return top-N recommendations using PCA-reduced content-based similarity (TF-IDF + PCA)."""
    sim = load_content_pca_similarity(df, _content_pca_sim_path, n_components=n_components)
    try:
        idx_map = load_index_map(_reviews_index_map_path)
        idx = idx_map.get(str(product_id))
        if idx is None:
            print(f"Product ID '{product_id}' not found in content PCA index map.")
            return []
    except Exception:
        try:
            idx = int(df[df['product_id'] == product_id].index[0])
        except Exception:
            print(f"Product ID '{product_id}' not found.")
            return []
    sim_row = np.array(sim[idx])
    sim_scores = list(enumerate(sim_row))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:N+1]
    product_indices = [i[0] for i in sim_scores]
    agg = aggregate_product_content(df)
    rec_ids = agg.iloc[product_indices]['product_id'].tolist()
    recommended_products = df[df['product_id'].isin(rec_ids)][['product_id', 'product_name', 'category', 'rating', 'discounted_price', 'img_link']]
    return recommended_products

# --- Sentiment-Based Recommendations ---
_sentiment_sim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../app/dataset/cosine_sim_sentiment.npy'))
_sentiment_scores_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../app/dataset/sentiment_scores.parquet'))

def aggregate_sentiment(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Load cached sentiment scores or compute them if missing.
    This function now prioritizes loading from a parquet cache.
    """
    # 1. Try to load from cache first
    if os.path.exists(_sentiment_scores_path):
        try:
            return pd.read_parquet(_sentiment_scores_path)
        except Exception as e:
            print(f"Warning: Could not load sentiment cache. Recomputing. Error: {e}")

    # 2. If cache is missing or fails, compute, save, and return
    print("Sentiment cache not found. Computing sentiment scores...")
    agg = aggregate_sentiment_parallel(df_input)
    try:
        os.makedirs(os.path.dirname(_sentiment_scores_path), exist_ok=True)
        agg.to_parquet(_sentiment_scores_path, index=False)
        print(f"Sentiment scores saved to cache: {_sentiment_scores_path}")
    except Exception as e:
        print(f"Warning: Failed to save sentiment cache. Error: {e}")
    return agg

def aggregate_sentiment_parallel(df_input: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentiment polarity for each product by averaging sentiment of all its reviews, using parallel processing."""
    def compute_single_sentiment(text):
        if not isinstance(text, str) or not text.strip():
            return 0.0
        return TextBlob(text).sentiment.polarity

    def process_reviews(review_content):
        if pd.isna(review_content):
            return 0.0
        reviews = str(review_content).split(',')
        if not reviews:
            return 0.0
        
        scores = Parallel(n_jobs=-1, backend='threading')(
            delayed(compute_single_sentiment)(r) for r in reviews if r.strip()
        )
        return np.mean(scores) if scores else 0.0

    # Create a DataFrame with product_id and review_content
    agg = df_input[['product_id', 'review_content']].copy().drop_duplicates(subset=['product_id']).reset_index(drop=True)
    
    # Apply the parallel sentiment processing
    agg['sentiment_score'] = agg['review_content'].apply(process_reviews)
    
    return agg[['product_id', 'sentiment_score']]

def build_and_save_sentiment_similarity(df_input: pd.DataFrame, path: str = _sentiment_sim_path):
    """Compute and save cosine similarity matrix based on sentiment scores."""
    agg = aggregate_sentiment(df_input)
    X = agg[['sentiment_score']].values
    sim = cosine_similarity(X)
    save_similarity_to_file(sim, path)
    # Save index map (product_id -> row index)
    index_map = {str(pid): int(i) for i, pid in enumerate(agg['product_id'].astype(str).tolist())}
    save_index_map(index_map, _reviews_index_map_path)  # reuse index map path
    return sim

def load_sentiment_similarity(df, path: str = _sentiment_sim_path):
    if not os.path.exists(path):
        return build_and_save_sentiment_similarity(df, path)
    return load_similarity_from_file(path)

def get_sentiment_recommendations(product_id, N=5):
    """Return top-N recommendations using sentiment-based similarity."""
    sim = load_sentiment_similarity(df, _sentiment_sim_path)
    try:
        idx_map = load_index_map(_reviews_index_map_path)
        idx = idx_map.get(str(product_id))
        if idx is None:
            print(f"Product ID '{product_id}' not found in sentiment index map.")
            return []
    except Exception:
        try:
            idx = int(df[df['product_id'] == product_id].index[0])
        except Exception:
            print(f"Product ID '{product_id}' not found.")
            return []
    sim_row = np.array(sim[idx])
    sim_scores = list(enumerate(sim_row))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:N+1]
    product_indices = [i[0] for i in sim_scores]
    
    # Use the cached sentiment scores, which are aligned with the similarity matrix
    agg = aggregate_sentiment(df)
    
    rec_ids = agg.iloc[product_indices]['product_id'].tolist()
    recommended_products = df[df['product_id'].isin(rec_ids)][['product_id', 'product_name', 'category', 'rating', 'discounted_price', 'img_link']]
    return recommended_products

# --- Topic Modeling-Based Recommendations (LDA on aggregated reviews) ---

_topic_sim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../app/dataset/cosine_sim_topic.npy'))

def build_and_save_topic_similarity(df_input: pd.DataFrame, path: str = _topic_sim_path, n_topics: int = 10):
    """Compute and save cosine similarity matrix based on LDA topic distributions of aggregated reviews."""
    agg = aggregate_reviews(df_input)
    # Use CountVectorizer for LDA
    vectorizer = CountVectorizer(stop_words='english', max_features=10000)
    X = vectorizer.fit_transform(agg['all_reviews'])
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    topic_distributions = lda.fit_transform(X)
    sim = cosine_similarity(topic_distributions)
    save_similarity_to_file(sim, path)
    # Save index map (product_id -> row index)
    index_map = {str(pid): int(i) for i, pid in enumerate(agg['product_id'].astype(str).tolist())}
    save_index_map(index_map, _reviews_index_map_path)
    return sim

def load_topic_similarity(df, path: str = _topic_sim_path, n_topics: int = 10):
    if not os.path.exists(path):
        return build_and_save_topic_similarity(df, path, n_topics=n_topics)
    return load_similarity_from_file(path)

def get_topic_recommendations(product_id, N=5, n_topics: int = 10):
    """Return top-N recommendations using topic modeling (LDA) on reviews."""
    sim = load_topic_similarity(df, _topic_sim_path, n_topics=n_topics)
    try:
        idx_map = load_index_map(_reviews_index_map_path)
        idx = idx_map.get(str(product_id))
        if idx is None:
            print(f"Product ID '{product_id}' not found in topic index map.")
            return []
    except Exception:
        try:
            idx = int(df[df['product_id'] == product_id].index[0])
        except Exception:
            print(f"Product ID '{product_id}' not found.")
            return []
    sim_row = np.array(sim[idx])
    sim_scores = list(enumerate(sim_row))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:N+1]
    product_indices = [i[0] for i in sim_scores]
    agg = aggregate_reviews(df)
    rec_ids = agg.iloc[product_indices]['product_id'].tolist()
    recommended_products = df[df['product_id'].isin(rec_ids)][['product_id', 'product_name', 'category', 'rating', 'discounted_price', 'img_link']]
    return recommended_products

def get_reviewer_overlap_recommendations(df, product_id, N=5):
    """Return top-N products reviewed by users who also reviewed the given product."""
    # Find users who reviewed the given product
    users = set(df[df['product_id'] == product_id]['user_id'])
    if not users:
        return pd.DataFrame()
    # Find all products reviewed by these users (excluding the current product)
    overlap = df[df['user_id'].isin(users) & (df['product_id'] != product_id)]
    # Count frequency of each product
    product_counts = overlap['product_id'].value_counts().head(N)
    if product_counts.empty:
        return pd.DataFrame()
    # Get product details for the top-N
    recs = df[df['product_id'].isin(product_counts.index)][['product_id', 'product_name', 'category', 'rating', 'discounted_price', 'img_link']].drop_duplicates()
    # Sort by frequency in product_counts
    recs['overlap_count'] = recs['product_id'].map(product_counts)
    recs = recs.sort_values('overlap_count', ascending=False).drop(columns=['overlap_count'])
    return recs

def get_hybrid_recommendations(df, product_id, N=5):
    """
    Return top-N hybrid recommendations by aggregating ranks from multiple recommenders:
    - get_recommendations_with_pca
    - get_review_recommendations
    - get_content_recommendations_pca
    - get_sentiment_recommendations
    - get_topic_recommendations
    - get_reviewer_overlap_recommendations
    Uses a simple Borda count (rank sum) approach.
    """
    # Get recommendations from each method (up to N*2 to allow for overlap)
    n_each = max(N * 2, 10)
    rec_lists = [
        get_recommendations_with_pca(df, product_id, N=n_each),
        get_review_recommendations(df, product_id, N=n_each),
        get_content_recommendations_pca(df, product_id, N=n_each),
        get_sentiment_recommendations(df, product_id, N=n_each),
        get_topic_recommendations(df, product_id, N=n_each),
        get_reviewer_overlap_recommendations(df, product_id, N=n_each),
    ]
    # Flatten to product_id lists, preserving order (rank)
    ranked_lists = []
    for recs in rec_lists:
        if hasattr(recs, 'product_id'):
            ids = recs['product_id'].tolist()
        elif isinstance(recs, list):
            ids = [r['product_id'] for r in recs if 'product_id' in r]
        else:
            ids = []
        ranked_lists.append(ids)
    # Borda count: assign points based on rank in each list
    from collections import Counter, defaultdict
    score = defaultdict(int)
    for ids in ranked_lists:
        for rank, pid in enumerate(ids):
            score[pid] += len(ids) - rank  # higher rank = more points
    # Sort by total score, exclude the current product
    sorted_pids = [pid for pid, _ in sorted(score.items(), key=lambda x: -x[1]) if pid != product_id]
    top_pids = sorted_pids[:N]
    # Get product details for the top-N
    recs = df[df['product_id'].isin(top_pids)][['product_id', 'product_name', 'category', 'rating', 'discounted_price', 'img_link']]
    # Preserve order
    recs['hybrid_score'] = recs['product_id'].map(score)
    recs = recs.set_index('product_id').loc[top_pids].reset_index()
    return recs

def get_weighted_hybrid_recommendations(df, product_id, N=5, weights=None):
    """
    Return top-N hybrid recommendations using user-defined weights for each recommender.
    - weights: dict mapping method name to weight (e.g., {"content_pca": 0.5, "review": 0.2, ...})
    Allowed keys: 'pca', 'review', 'content_pca', 'sentiment', 'topic', 'reviewer_overlap'
    """
    if weights is None:
        # Default: equal weights
        weights = {
            'basic_cosine': 0.0502,
            'pca_features': 0.0430,
            'content_tfidf': 0.0098,
            'content_pca': 0.0007,
            'review_text': 0.1902,
            'sentiment': 0.1712,
            'topic_lda': 0.0109,
            'reviewer_overlap': 0.0750,
            'knn_numeric': 0.3903,
            'svd_collaborative': 0.0587
        }
    # Normalize weights
    total = sum(weights.values())
    if total == 0:
        raise ValueError("At least one weight must be > 0")
    norm_weights = {k: v / total for k, v in weights.items()}

    n_each = max(N * 2, 10)
    # Get recommendations from each method
    rec_methods = {
        'pca': get_recommendations_with_pca(df, product_id, N=n_each),
        'review': get_review_recommendations(df, product_id, N=n_each),
        'content_pca': get_content_recommendations_pca(df, product_id, N=n_each),
        'sentiment': get_sentiment_recommendations(df, product_id, N=n_each),
        'topic': get_topic_recommendations(df, product_id, N=n_each),
        'reviewer_overlap': get_reviewer_overlap_recommendations(df, product_id, N=n_each),
    }
    # Build ranked lists
    ranked_lists = {}
    for key, recs in rec_methods.items():
        if hasattr(recs, 'product_id'):
            ids = recs['product_id'].tolist()
        elif isinstance(recs, list):
            ids = [r['product_id'] for r in recs if 'product_id' in r]
        else:
            ids = []
        ranked_lists[key] = ids
    # Weighted Borda count
    from collections import defaultdict
    score = defaultdict(float)
    for key, ids in ranked_lists.items():
        w = norm_weights.get(key, 0)
        for rank, pid in enumerate(ids):
            score[pid] += w * (len(ids) - rank)
    # Sort by total score, exclude the current product
    sorted_pids = [pid for pid, _ in sorted(score.items(), key=lambda x: -x[1]) if pid != product_id]
    top_pids = sorted_pids[:N]
    recs = df[df['product_id'].isin(top_pids)][['product_id', 'product_name', 'category', 'rating', 'discounted_price', 'img_link']]
    recs['weighted_hybrid_score'] = recs['product_id'].map(score)
    recs = recs.set_index('product_id').loc[top_pids].reset_index()
    return recs

# --- ML-based Trending Products ---
def get_trending_products_ml(df, N=10, weights=None):
    """
    Return top-N trending products using a hybrid ML-inspired score:
    - Number of positive reviews (rating >= 4)
    - Average rating
    - Sentiment score (from aggregate_sentiment)
    - (Optionally: recency if timestamp is present)
    All features are normalized and combined with weights.
    """
    # Defensive: check required columns
    required_cols = ['product_id', 'rating', 'review_content']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in product DataFrame.")

    # Create a working copy of the DataFrame
    df_trend = df.copy()

    # Count positive reviews (rating >= 4)
    def count_positive_reviews(row):
        if pd.isna(row['rating']):
            return 0
        # If multiple ratings, split and count
        if isinstance(row['rating'], list):
            ratings = row['rating']
        else:
            ratings = [r for r in str(row['rating']).split(',') if r.strip()]
        count = 0
        for r in ratings:
            try:
                if float(r) >= 4:
                    count += 1
            except Exception:
                continue  # skip non-numeric
        return count
    df_trend['positive_review_count'] = df_trend.apply(count_positive_reviews, axis=1)
    # Average rating
    def avg_rating(row):
        if pd.isna(row['rating']):
            return 0.0
        if isinstance(row['rating'], list):
            ratings = row['rating']
        else:
            ratings = [r for r in str(row['rating']).split(',') if r.strip()]
        vals = []
        for r in ratings:
            try:
                vals.append(float(r))
            except Exception:
                continue
        return np.mean(vals) if vals else 0.0
    df_trend['avg_rating'] = df_trend.apply(avg_rating, axis=1)
    # Sentiment score
    sentiment_df = aggregate_sentiment(df_trend)
    df_trend = df_trend.merge(sentiment_df, on='product_id', how='left')
    # Normalize features
    def norm(col):
        arr = df_trend[col].values.astype(float)
        minv, maxv = np.min(arr), np.max(arr)
        if maxv - minv == 0:
            return np.zeros_like(arr)
        return (arr - minv) / (maxv - minv)
    df_trend['norm_pos'] = norm('positive_review_count')
    df_trend['norm_rating'] = norm('avg_rating')
    df_trend['norm_sent'] = norm('sentiment_score')
    # Weights for each feature
    if weights is None:
        weights = {'pos': 0.5, 'rating': 0.3, 'sent': 0.2}
    # Compute trending score
    df_trend['trending_score'] = (
        weights['pos'] * df_trend['norm_pos'] +
        weights['rating'] * df_trend['norm_rating'] +
        weights['sent'] * df_trend['norm_sent']
    )
    # Sort and return top-N
    df_trend = df_trend.drop_duplicates(subset="product_id")
    df_trend = df_trend.sort_values('trending_score', ascending=False)
    cols = ['product_id', 'product_name', 'category', 'avg_rating', 'positive_review_count', 'sentiment_score', 'trending_score', 'discounted_price', 'img_link']
    return df_trend[cols].head(N)

def prepare_features_for_knn(df, feature_cols=None):
    """
    Cleans and prepares feature matrix for KNN recommendations.
    Returns cleaned DataFrame and feature matrix (numpy array).
    """
    if feature_cols is None:
        feature_cols = ['discounted_price', 'actual_price', 'rating_count', 'discount_percentage', 'rating']
    feature_cols = [c for c in feature_cols if c in df.columns]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[feature_cols] = df[feature_cols].fillna(0)
    X = df[feature_cols].values
    return df, X

def get_knn_recommendations(df, X, product_idx, n=5):
    """
    Fits a KNN model and returns n similar products for a given product index.
    Returns a DataFrame with product details.
    """
    
    knn = NearestNeighbors(n_neighbors=n+1, metric='cosine', algorithm='brute')
    knn.fit(X)
    query_vector = X[product_idx].reshape(1, -1)
    distances, indices = knn.kneighbors(query_vector, n_neighbors=n+1)
    similar_indices = indices.flatten()[1:]  # Exclude the product itself
    return df.iloc[similar_indices][['product_id', 'product_name', 'rating', 'discounted_price']]

def svd_product_recommendations(df, query_product_id, n=5):
    """
    Given a DataFrame with columns: user_id, product_id, rating, returns top-n similar products for a given product_id using SVD matrix factorization and cosine similarity.
    """

    # Create user-item matrix
    user_item_matrix = df.pivot_table(index='user_id', columns='product_id', values='rating', fill_value=0)
    # Center the data
    user_ratings_mean = np.mean(user_item_matrix.values, axis=1)
    R_demeaned = user_item_matrix.values - user_ratings_mean.reshape(-1, 1)
    # Run SVD
    U, sigma, Vt = svds(R_demeaned, k=20)
    sigma = np.diag(sigma)
    # Reconstruct predicted ratings
    predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    pred_df = pd.DataFrame(predicted_ratings, index=user_item_matrix.index, columns=user_item_matrix.columns)
    # Query for similar products by product_id
    if query_product_id not in user_item_matrix.columns:
        raise ValueError(f"Product ID {query_product_id} not found in the user-item matrix.")
    product_idx = list(user_item_matrix.columns).index(query_product_id)
    product_vector = pred_df.iloc[:, product_idx].values.reshape(1, -1)
    all_product_vectors = pred_df.values.T  # shape: (num_products, num_users)
    similarities = cosine_similarity(product_vector, all_product_vectors).flatten()
    similar_indices = similarities.argsort()[::-1][1:n+1]
    similar_product_ids = [user_item_matrix.columns[i] for i in similar_indices]
    # Return DataFrame with product details
    return df[df['product_id'].isin(similar_product_ids)][['product_id', 'product_name', 'rating', 'discounted_price']].drop_duplicates(subset=['product_id'])

# Load data and models
df = None
cosine_sim = None
cosine_sim_content = None

def load_all_models(data_path='../app/dataset/'):
    """Loads all models and data artifacts."""
    global df, cosine_sim, cosine_sim_content, cosine_sim_pca, cosine_sim_content_pca, cosine_sim_reviews, cosine_sim_sentiment, cosine_sim_topic, index_map, pca, preprocessor
    
    try:
        df = pd.read_parquet(os.path.join(data_path, 'amazon_clean.parquet'))
        
        # Load similarity matrices
        cosine_sim = load_similarity_from_file(os.path.join(data_path, 'cosine_sim.npy'))
        cosine_sim_content = load_similarity_from_file(os.path.join(data_path, 'cosine_sim_content.npy'))
        cosine_sim_pca = load_similarity_from_file(os.path.join(data_path, 'cosine_sim_pca.npy'))
        cosine_sim_content_pca = load_similarity_from_file(os.path.join(data_path, 'cosine_sim_content_pca.npy'))
        cosine_sim_reviews = load_similarity_from_file(os.path.join(data_path, 'cosine_sim_reviews.npy'))
        cosine_sim_sentiment = load_similarity_from_file(os.path.join(data_path, 'cosine_sim_sentiment.npy'))
        cosine_sim_topic = load_similarity_from_file(os.path.join(data_path, 'cosine_sim_topic.npy'))
        
        # Load index map
        index_map = load_index_map(os.path.join(data_path, 'index_map.json'))
        
        # Load PCA and preprocessor
        pca = load_pca(os.path.join(data_path, 'pca.joblib'))
        preprocessor = load_preprocessor(os.path.join(data_path, 'preprocessor.joblib'))
        
        print("All models and data loaded successfully.")
        return df
    except FileNotFoundError as e:
        print(f"Error loading files: {e}. Please ensure all artifact paths are correct.")
        return None

def _indices_to_recommendations(df_local: pd.DataFrame, product_indices: list, index_map_path: str = _index_map_path, id_col: str = 'product_id') -> pd.DataFrame:
    """
    Convert a list of integer indices (rows in a similarity matrix) into a
    DataFrame of product details from `df_local`, preserving order. This handles
    cases where the similarity matrix index ordering differs from `df_local`.

    Strategy:
    - If an index map file exists, load it and reverse it to map index -> product_id.
      Use these product_ids to select rows from df_local in the same order.
    - Otherwise, fall back to using iloc but clamp indices to valid range to avoid
      positional indexer 'out-of-bounds' errors.
    """
    # Try to load index_map and reverse it
    try:
        idx_map = load_index_map(index_map_path)
        rev_map = {int(v): str(k) for k, v in idx_map.items()}
        rec_ids = [rev_map[i] for i in product_indices if i in rev_map]
        if not rec_ids:
            # fallback to iloc clamping
            raise KeyError("No matching product_ids found in index_map for given indices")
        # Use set_index for deterministic selection while preserving order
        df_by_id = df_local.set_index(id_col)
        # filter existing ids and preserve order
        existing_ids = [pid for pid in rec_ids if pid in df_by_id.index]
        if not existing_ids:
            # fallback to iloc clamping
            raise KeyError("Mapped product_ids not present in provided df")
        recs = df_by_id.loc[existing_ids].reset_index()
        return recs
    except Exception:
        # Safe iloc fallback: clamp indices to valid range
        if len(df_local) == 0:
            return pd.DataFrame()
        max_idx = len(df_local) - 1
        safe_idxs = [i for i in product_indices if 0 <= i <= max_idx]
        if not safe_idxs:
            return pd.DataFrame()
        # Remove duplicates while preserving order
        seen = set()
        safe_idxs_uniq = []
        for i in safe_idxs:
            if i not in seen:
                seen.add(i)
                safe_idxs_uniq.append(i)
        try:
            recs = df_local.iloc[safe_idxs_uniq][['product_id', 'product_name', 'category', 'rating', 'discounted_price', 'img_link']]
            return recs
        except Exception:
            return pd.DataFrame()

def get_recommendations(*args, **kwargs):
    """
    Flexible wrapper for getting N product recommendations based on cosine similarity.

    Supports calling conventions:
      - get_recommendations(df, product_id, N=...)
      - get_recommendations(product_id, N=...)
      - get_recommendations(product_id, n) (positional)
    """
    # Defaults
    df_local = None
    product_id = None
    N = kwargs.get('N', kwargs.get('n', 5))

    # Parse positional args
    if len(args) == 1:
        # Could be (product_id,) using module-level df
        product_id = args[0]
        df_local = globals().get('df')
    elif len(args) >= 2:
        df_local = args[0]
        product_id = args[1]
        if len(args) >= 3:
            # third positional arg may be N
            N = args[2]

    # If product_id provided via kwargs
    if product_id is None and 'product_id' in kwargs:
        product_id = kwargs['product_id']
    # If df provided via kwargs
    if df_local is None and 'df' in kwargs:
        df_local = kwargs['df']

    # Final sanity
    if product_id is None:
        raise ValueError('product_id must be provided')

    if df_local is None:
        print("Data or model not loaded. Please run load_all_models() first.")
        return pd.DataFrame()

    # Ensure cosine_sim is loaded
    global cosine_sim
    if cosine_sim is None:
        print("Cosine similarity matrix not loaded. Please run load_all_models() first.")
        return pd.DataFrame()

    # Resolve index using index_map if available
    try:
        idx_map = load_index_map()
        idx = idx_map.get(str(product_id))
        if idx is None:
            print(f"Product ID '{product_id}' not found in index map.")
            return pd.DataFrame()
    except Exception:
        # fallback: try to find in df_local
        try:
            idx = int(df_local[df_local['product_id'] == product_id].index[0])
        except Exception:
            print(f"Product ID '{product_id}' not found.")
            return pd.DataFrame()

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:int(N)+1]
    product_indices = [i[0] for i in sim_scores]

    # Return the top N most similar product details
    recommended_products = _indices_to_recommendations(df_local, product_indices)
    return recommended_products

def get_recommendations_with_pca(df, product_id, N=5):
    """
    Get N product recommendations based on cosine similarity after PCA.
    
    Args:
        df (pd.DataFrame): The DataFrame containing product information.
        product_id (str): The product_id for which to find recommendations.
        N (int): The number of recommendations to return.
        
    Returns:
        pd.DataFrame: A DataFrame with the top N recommended products.
    """
    if df is None or cosine_sim_pca is None:
        print("Data or model not loaded. Please run load_all_models() first.")
        return pd.DataFrame()

    try:
        idx_map = load_index_map()
        idx = idx_map.get(str(product_id))
        if idx is None:
            print(f"Product ID '{product_id}' not found in index map.")
            return pd.DataFrame()
    except Exception:
        try:
            idx = int(df[df['product_id'] == product_id].index[0])
        except Exception:
            print(f"Product ID '{product_id}' not found.")
            return pd.DataFrame()

    idx = index_map[product_id]
    sim_scores = list(enumerate(cosine_sim_pca[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:N+1]
    product_indices = [i[0] for i in sim_scores]
    
    # Return the top N most similar product details
    recommended_products = _indices_to_recommendations(df, product_indices)
    return recommended_products

def get_content_recommendations(df, product_id, N=5):
    """
    Get N product recommendations based on content similarity.
    
    Args:
        df (pd.DataFrame): The DataFrame containing product information.
        product_id (str): The product_id for which to find recommendations.
        N (int): The number of recommendations to return.
        
    Returns:
        pd.DataFrame: A DataFrame with the top N recommended products.
    """
    if df is None or cosine_sim_content is None:
        print("Data or model not loaded. Please run load_all_models() first.")
        return pd.DataFrame()

    try:
        idx_map = load_index_map()
        idx = idx_map.get(str(product_id))
        if idx is None:
            print(f"Product ID '{product_id}' not found in index map.")
            return pd.DataFrame()
    except Exception:
        try:
            idx = int(df[df['product_id'] == product_id].index[0])
        except Exception:
            print(f"Product ID '{product_id}' not found.")
            return pd.DataFrame()

    idx = index_map[product_id]
    sim_scores = list(enumerate(cosine_sim_content[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:N+1]
    product_indices = [i[0] for i in sim_scores]
    
    # Return the top N most similar product details
    recommended_products = _indices_to_recommendations(df, product_indices)
    return recommended_products

def get_content_recommendations_pca(df, product_id, N=5):
    """
    Get N product recommendations based on content similarity after PCA.
    
    Args:
        df (pd.DataFrame): The DataFrame containing product information.
        product_id (str): The product_id for which to find recommendations.
        N (int): The number of recommendations to return.
        
    Returns:
        pd.DataFrame: A DataFrame with the top N recommended products.
    """
    if df is None or cosine_sim_content_pca is None:
        print("Data or model not loaded. Please run load_all_models() first.")
        return pd.DataFrame()

    try:
        idx_map = load_index_map()
        idx = idx_map.get(str(product_id))
        if idx is None:
            print(f"Product ID '{product_id}' not found in index map.")
            return pd.DataFrame()
    except Exception:
        try:
            idx = int(df[df['product_id'] == product_id].index[0])
        except Exception:
            print(f"Product ID '{product_id}' not found.")
            return pd.DataFrame()

    idx = index_map[product_id]
    sim_scores = list(enumerate(cosine_sim_content_pca[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:N+1]
    product_indices = [i[0] for i in sim_scores]
    
    # Return the top N most similar product details
    recommended_products = _indices_to_recommendations(df, product_indices)
    return recommended_products

def get_review_recommendations(df, product_id, N=5):
    """
    Get N product recommendations based on review text similarity.
    
    Args:
        df (pd.DataFrame): The DataFrame containing product information.
        product_id (str): The product_id for which to find recommendations.
        N (int): The number of recommendations to return.
        
    Returns:
        pd.DataFrame: A DataFrame with the top N recommended products.
    """
    if df is None or cosine_sim_reviews is None:
        print("Data or model not loaded. Please run load_all_models() first.")
        return pd.DataFrame()

    try:
        idx_map = load_index_map()
        idx = idx_map.get(str(product_id))
        if idx is None:
            print(f"Product ID '{product_id}' not found in index map.")
            return pd.DataFrame()
    except Exception:
        try:
            idx = int(df[df['product_id'] == product_id].index[0])
        except Exception:
            print(f"Product ID '{product_id}' not found.")
            return pd.DataFrame()

    idx = index_map[product_id]
    sim_scores = list(enumerate(cosine_sim_reviews[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:N+1]
    product_indices = [i[0] for i in sim_scores]
    
    # Return the top N most similar product details
    recommended_products = _indices_to_recommendations(df, product_indices)
    return recommended_products

def get_sentiment_recommendations(df, product_id, N=5):
    """
    Get N product recommendations based on sentiment similarity.
    
    Args:
        df (pd.DataFrame): The DataFrame containing product information.
        product_id (str): The product_id for which to find recommendations.
        N (int): The number of recommendations to return.
        
    Returns:
        pd.DataFrame: A DataFrame with the top N recommended products.
    """
    if df is None or cosine_sim_sentiment is None:
        print("Data or model not loaded. Please run load_all_models() first.")
        return pd.DataFrame()

    try:
        idx_map = load_index_map()
        idx = idx_map.get(str(product_id))
        if idx is None:
            print(f"Product ID '{product_id}' not found in index map.")
            return pd.DataFrame()
    except Exception:
        try:
            idx = int(df[df['product_id'] == product_id].index[0])
        except Exception:
            print(f"Product ID '{product_id}' not found.")
            return pd.DataFrame()

    idx = index_map[product_id]
    sim_scores = list(enumerate(cosine_sim_sentiment[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:N+1]
    product_indices = [i[0] for i in sim_scores]
    
    # Return the top N most similar product details
    recommended_products = _indices_to_recommendations(df, product_indices)
    return recommended_products

def get_topic_recommendations(df, product_id, N=5):
    """
    Get N product recommendations based on topic modeling similarity.
    
    Args:
        df (pd.DataFrame): The DataFrame containing product information.
        product_id (str): The product_id for which to find recommendations.
        N (int): The number of recommendations to return.
        
    Returns:
        pd.DataFrame: A DataFrame with the top N recommended products.
    """
    if df is None or cosine_sim_topic is None:
        print("Data or model not loaded. Please run load_all_models() first.")
        return pd.DataFrame()

    try:
        idx_map = load_index_map()
        idx = idx_map.get(str(product_id))
        if idx is None:
            print(f"Product ID '{product_id}' not found in index map.")
            return pd.DataFrame()
    except Exception:
        try:
            idx = int(df[df['product_id'] == product_id].index[0])
        except Exception:
            print(f"Product ID '{product_id}' not found.")
            return pd.DataFrame()

    idx = index_map[product_id]
    sim_scores = list(enumerate(cosine_sim_topic[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:N+1]
    product_indices = [i[0] for i in sim_scores]
    
    # Return the top N most similar product details
    recommended_products = _indices_to_recommendations(df, product_indices)
    return recommended_products

def get_reviewer_overlap_recommendations(df, product_id, N=5):
    """
    Finds products that have been reviewed by the same users who reviewed the given product.
    
    Args:
        df (pd.DataFrame): The DataFrame containing product information.
        product_id (str): The ID of the product to get recommendations for.
        N (int): The number of recommendations to return.
        
    Returns:
        pd.DataFrame: A DataFrame of recommended products.
    """
    if df is None:
        print("Data not loaded. Please run load_all_models() first.")
        return pd.DataFrame()

    # Find all users who reviewed the target product
    reviewers = df[df['product_id'] == product_id]['user_id'].unique()
    if not reviewers.any():
        return pd.DataFrame()
    
    # Find all products reviewed by these users, excluding the target product
    recommended_products = df[df['user_id'].isin(reviewers) & (df['product_id'] != product_id)]
    
    # Count the number of reviews for each product
    product_counts = recommended_products['product_id'].value_counts()
    
    # Get the top N recommendations
    top_n_products = product_counts.head(N)
    
    # Retrieve full product details
    recommended_details = df[df['product_id'].isin(top_n_products.index)][['product_id', 'product_name', 'category', 'rating', 'discounted_price', "img_link"]].drop_duplicates().set_index('product_id')
    
    if top_n_products.index.empty:
        return pd.DataFrame()
        
    recommended_details = recommended_details.loc[top_n_products.index]
    
    return recommended_details.reset_index()

def get_weighted_hybrid_recommendations(df, product_id, N=5, weights=None):
    """
    10-Model Hybrid Engine: Combines every strategy into one weighted score.
    Synchronized with 00_Eval.ipynb randomized search keys.
    """
    if df is None:
        print("Data not loaded. Please run load_all_models() first.")
        return pd.DataFrame()

    # Define standard keys used for weights
    model_keys = [
        'basic_cosine', 'pca_features', 'content_tfidf', 'content_pca', 
        'review_text', 'sentiment', 'topic_lda', 'reviewer_overlap', 
        'knn_numeric', 'svd_collaborative'
    ]

    if weights is None:
        # Default: equal weights
        weights = {k: 1.0/len(model_keys) for k in model_keys}

    # Normalize weights so they sum to 1.0
    total = sum(weights.values())
    norm_weights = {k: v / total for k, v in weights.items()}

    # Prepare specific data for KNN
    try:
        df_knn, X_knn = prepare_features_for_knn(df)
        p_idx = df[df['product_id'] == product_id].index[0]
    except Exception:
        p_idx = None

    # Retrieve candidates from all 10 specialized experts
    n_pool = N * 2
    recs_to_combine = {
        'basic_cosine': get_recommendations(df, product_id, N=n_pool),
        'pca_features': get_recommendations_with_pca(df, product_id, N=n_pool),
        'content_tfidf': get_content_recommendations(df, product_id, N=n_pool),
        'content_pca': get_content_recommendations_pca(df, product_id, N=n_pool),
        'review_text': get_review_recommendations(df, product_id, N=n_pool),
        'sentiment': get_sentiment_recommendations(df, product_id, N=n_pool),
        'topic_lda': get_topic_recommendations(df, product_id, N=n_pool),
        'reviewer_overlap': get_reviewer_overlap_recommendations(df, product_id, N=n_pool),
    }
    
    # Add KNN and SVD if indices were found
    if p_idx is not None:
        recs_to_combine['knn_numeric'] = get_knn_recommendations(df, X_knn, p_idx, n=n_pool)
        recs_to_combine['svd_collaborative'] = svd_product_recommendations(df, product_id, n=n_pool)

    # Weighted Borda count scoring to avoid KeyError
    from collections import defaultdict
    final_scores = defaultdict(float)
    
    for method, rec_df in recs_to_combine.items():
        if rec_df is not None and not rec_df.empty:
            w = norm_weights.get(method, 0) # Uses .get() to safely handle missing keys
            pids = rec_df['product_id'].tolist()
            for rank, pid in enumerate(pids):
                # Points = weight * (pool_size - current_rank)
                final_scores[pid] += w * (len(pids) - rank)

    # Sort and return results
    sorted_pids = [pid for pid, _ in sorted(final_scores.items(), key=lambda x: -x[1]) if pid != product_id]
    top_pids = sorted_pids[:N]
    
    return df[df['product_id'].isin(top_pids)].copy().assign(
        hybrid_score=lambda d: d['product_id'].map(final_scores)
    ).sort_values('hybrid_score', ascending=False).reset_index(drop=True)


# This is the main function that will be called by the FastAPI app
def find_similar_products(df, product_id: str, N: int = 5, method: str = 'cosine', weights: dict = None):
    """
    Main function to find similar products using a specified method.
    
    Args:
        df (pd.DataFrame): The DataFrame containing product information.
        product_id (str): The product ID to get recommendations for.
        N (int): The number of recommendations to return.
        method (str): The recommendation method to use.
        weights (dict): A dictionary with weights for each model (for hybrid method).
        
    Returns:
        pd.DataFrame: A DataFrame of recommended products.
    """
    if df is None:
        print("Data not loaded. Please run load_all_models() first.")
        return pd.DataFrame()

    if method == 'cosine':
        return get_recommendations(df, product_id, N)
    elif method == 'pca':
        return get_recommendations_with_pca(df, product_id, N)
    elif method == 'content':
        return get_content_recommendations(df, product_id, N)
    elif method == 'content_pca':
        return get_content_recommendations_pca(df, product_id, N)
    elif method == 'reviews':
        return get_review_recommendations(df, product_id, N)
    elif method == 'sentiment':
        return get_sentiment_recommendations(df, product_id, N)
    elif method == 'topic':
        return get_topic_recommendations(df, product_id, N)
    elif method == 'reviewer_overlap':
        return get_reviewer_overlap_recommendations(df, product_id, N)
    elif method == 'hybrid':
        return get_weighted_hybrid_recommendations(df, product_id, N, weights)
    else:
        raise ValueError("Invalid method specified. Choose from: 'cosine', 'pca', 'content', 'content_pca', 'reviews', 'sentiment', 'topic', 'reviewer_overlap', 'hybrid'")

if __name__ == '__main__':
    # This block is for testing purposes
    df = load_all_models()
    if df is not None:
        test_product_id = df['product_id'].iloc[0]  # Get a valid product_id for testing
        print(f"Testing with product ID: {test_product_id}")
        
        # Test each recommendation method
        methods_to_test = ['cosine', 'pca', 'content', 'content_pca', 'reviews', 'sentiment', 'topic', 'reviewer_overlap', 'hybrid']
        for method in methods_to_test:
            print(f"\n--- Testing method: {method} ---")
            recommendations = find_similar_products(df, test_product_id, N=5, method=method)
            print(recommendations)