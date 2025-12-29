import pandas as pd
from sklearn.linear_model import LinearRegression
import os
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from pathlib import Path
import json
import pymongo
from pymongo import MongoClient
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA, LatentDirichletAllocation

# Get the absolute paths to dataset files using pathlib for clarity and cross-platform correctness
_data_dir = Path(__file__).parent.joinpath('../app/dataset').resolve()
_abs_csv_path = _data_dir.joinpath('amazon.csv')
product_features_path = _data_dir.joinpath('product_features.csv')

# MongoDB connection settings
MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://raquelbastian_db_user:oXu3M164d7HEwcVL@capstone.jucteam.mongodb.net")
MONGO_DB = os.environ.get("MONGO_DB", "capstone")
MONGO_COLLECTION = os.environ.get("MONGO_COLLECTION", "products")

def load_products_from_mongodb():
    """Load products from MongoDB, clean currency/percent strings, and return as DataFrame."""
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        collection = db[MONGO_COLLECTION]
        products = list(collection.find())
        if not products:
            raise ValueError("No products found in MongoDB.")
        
        for p in products:
            p.pop('_id', None)
        
        df_mongo = pd.DataFrame(products)

        # --- DATA CLEANING STEP ---
        # 1. Clean Discounted Price: remove ₹ and commas
        if 'discounted_price' in df_mongo.columns:
            df_mongo['discounted_price'] = df_mongo['discounted_price'].astype(str).str.replace('₹', '').str.replace(',', '')
            df_mongo['discounted_price'] = pd.to_numeric(df_mongo['discounted_price'], errors='coerce').fillna(0)

        # 2. Clean Actual Price: remove ₹ and commas
        if 'actual_price' in df_mongo.columns:
            df_mongo['actual_price'] = df_mongo['actual_price'].astype(str).str.replace('₹', '').str.replace(',', '')
            df_mongo['actual_price'] = pd.to_numeric(df_mongo['actual_price'], errors='coerce').fillna(0)

        # 3. Clean Discount Percentage: remove %
        if 'discount_percentage' in df_mongo.columns:
            df_mongo['discount_percentage'] = df_mongo['discount_percentage'].astype(str).str.replace('%', '')
            df_mongo['discount_percentage'] = pd.to_numeric(df_mongo['discount_percentage'], errors='coerce').fillna(0)
        
        # 4. Clean Rating: ensure it is a float
        if 'rating' in df_mongo.columns:
            df_mongo['rating'] = pd.to_numeric(df_mongo['rating'], errors='coerce').fillna(0)

        return df_mongo
    except Exception as e:
        print(f"MongoDB load and clean failed: {e}")
        return None

# Try to load from MongoDB only. Do not fallback to CSV.
df = load_products_from_mongodb()
if df is not None:
    print("Loaded products from MongoDB.")
else:
    raise RuntimeError("Failed to load products from MongoDB. Please check your MongoDB connection and data.")

# Note: product features will be loaded lazily via load_product_features()

from sklearn.metrics.pairwise import cosine_similarity

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
        from sklearn.metrics.pairwise import cosine_similarity
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
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=sparse)
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
    """Load a saved preprocessor (joblib file)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessor file not found: {path}")
    return joblib.load(path)


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

def get_recommendations(product_id, N=5):
    # Get the index of the product that matches the product_id
    matrix = build_similarity_matrix()

    # Try to use index map if available for consistent mapping
    try:
        idx_map = load_index_map()
        idx = idx_map.get(str(product_id))
        if idx is None:
            print(f"Product ID '{product_id}' not found in index map.")
            return []
    except Exception:
        # fallback: try to find in df
        try:
            idx = int(df[df['product_id'] == product_id].index[0])
        except Exception:
            print(f"Product ID '{product_id}' not found.")
            return []

    # If matrix is memmaped, ensure indexing yields an array
    sim_row = np.array(matrix[idx])

    # Get the pairwise similarity scores of all products with that product
    sim_scores = list(enumerate(sim_row))

    # Sort the products based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the N most similar products (excluding itself)
    sim_scores = sim_scores[1:N+1]

    product_indices = [i[0] for i in sim_scores]

    # Return the top N most similar product details
    recommended_products = df.iloc[product_indices][['product_id', 'product_name', 'category', 'rating', 'discounted_price', "img_link"]]
    return recommended_products

print("Recommendation function 'get_recommendations' updated to use reduced feature set.")

# New: PCA-based dimensionality reduction + PCA-backed similarity matrix
# These functions add an alternative similarity artifact that computes cosine
# similarity on PCA-reduced encoded features and does not overwrite the
# existing cosine_sim.npy produced by the original pipeline.

_pca_artifact_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../app/dataset/pca.joblib'))
_abs_sim_cache_pca = os.path.abspath(os.path.join(os.path.dirname(__file__), '../app/dataset/cosine_sim_pca.npy'))


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
        from sklearn.metrics.pairwise import cosine_similarity
        mat = cosine_similarity(df_red.values)
        # persist
        save_similarity_to_file(mat, _abs_sim_cache_pca)
        return mat


def get_recommendations_with_pca(product_id, N=5, n_components: int = 50):
    """Return top-N recommendations using PCA-reduced cosine similarity.

    This mirrors `get_recommendations` but uses the PCA-backed similarity matrix.
    """
    # Load or build the PCA-backed similarity matrix
    try:
        matrix = build_similarity_matrix_with_pca(n_components=n_components)
    except Exception as e:
        print(f"Failed to build or load PCA-backed similarity matrix: {e}")
        return []

    # Resolve index using index_map if available
    try:
        idx_map = load_index_map()
        idx = idx_map.get(str(product_id))
        if idx is None:
            print(f"Product ID '{product_id}' not found in index map.")
            return []
    except Exception:
        try:
            idx = int(df[df['product_id'] == product_id].index[0])
        except Exception:
            print(f"Product ID '{product_id}' not found.")
            return []

    sim_row = np.array(matrix[idx])
    sim_scores = list(enumerate(sim_row))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:N+1]
    product_indices = [i[0] for i in sim_scores]

    recommended_products = df.iloc[product_indices][['product_id', 'product_name', 'category', 'rating', 'discounted_price', 'img_link']]
    return recommended_products

# end of PCA additions

# --- Review-based Similarity (TF-IDF on aggregated reviews) ---
from sklearn.feature_extraction.text import TfidfVectorizer

_reviews_sim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../app/dataset/cosine_sim_reviews.npy'))
_reviews_index_map_path = _index_map_path  # reuse index map

_reviews_agg_cache = None
_reviews_sim_cache = None

def aggregate_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all reviews for each product into a single string."""
    if 'review_content' not in df.columns:
        raise ValueError("review_content column not found in DataFrame")
    # Some products have comma-separated reviews in a single row
    def join_reviews(row):
        if pd.isna(row['review_content']):
            return ''
        # If already a list, join; else, split by comma and join
        if isinstance(row['review_content'], list):
            return ' '.join(row['review_content'])
        return ' '.join(str(row['review_content']).split(','))
    agg = df[['product_id', 'review_content']].copy()
    agg['all_reviews'] = agg.apply(join_reviews, axis=1)
    return agg[['product_id', 'all_reviews']]

def build_and_save_review_similarity(df: pd.DataFrame, path: str = _reviews_sim_path):
    """Compute and save cosine similarity matrix based on aggregated reviews."""
    agg = aggregate_reviews(df)
    tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
    X = tfidf.fit_transform(agg['all_reviews'])
    sim = cosine_similarity(X)
    save_similarity_to_file(sim, path)
    # Save index map (product_id -> row index)
    index_map = {str(pid): int(i) for i, pid in enumerate(agg['product_id'].astype(str).tolist())}
    save_index_map(index_map, _reviews_index_map_path)
    return sim

def load_review_similarity(path: str = _reviews_sim_path):
    if not os.path.exists(path):
        # Build if missing
        return build_and_save_review_similarity(df, path)
    return load_similarity_from_file(path)

def get_review_recommendations(product_id, N=5):
    """Return top-N recommendations using review-based similarity."""
    sim = load_review_similarity()
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
def aggregate_product_content(df: pd.DataFrame) -> pd.DataFrame:
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
    agg = df[['product_id', 'product_name', 'category', 'about_product', 'review_title', 'review_content']].copy()
    agg['all_content'] = agg.apply(join_fields, axis=1)
    return agg[['product_id', 'all_content']]

_content_sim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../app/dataset/cosine_sim_content.npy'))

def build_and_save_content_similarity(df: pd.DataFrame, path: str = _content_sim_path):
    """Compute and save cosine similarity matrix based on all product content."""
    agg = aggregate_product_content(df)
    tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
    X = tfidf.fit_transform(agg['all_content'])
    sim = cosine_similarity(X)
    save_similarity_to_file(sim, path)
    # Save index map (product_id -> row index)
    index_map = {str(pid): int(i) for i, pid in enumerate(agg['product_id'].astype(str).tolist())}
    save_index_map(index_map, _reviews_index_map_path)  # reuse index map path
    return sim

def load_content_similarity(path: str = _content_sim_path):
    if not os.path.exists(path):
        return build_and_save_content_similarity(df, path)
    return load_similarity_from_file(path)

def get_content_recommendations(product_id, N=5):
    """Return top-N recommendations using content-based similarity (all text fields)."""
    sim = load_content_similarity()
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
from sklearn.decomposition import PCA

_content_pca_sim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../app/dataset/cosine_sim_content_pca.npy'))


def build_and_save_content_pca_similarity(df: pd.DataFrame, path: str = _content_pca_sim_path, n_components: int = 50):
    """Compute and save cosine similarity matrix based on PCA-reduced TF-IDF of all product content."""
    agg = aggregate_product_content(df)
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

def load_content_pca_similarity(path: str = _content_pca_sim_path, n_components: int = 50):
    if not os.path.exists(path):
        return build_and_save_content_pca_similarity(df, path, n_components=n_components)
    return load_similarity_from_file(path)

def get_content_recommendations_pca(product_id, N=5, n_components: int = 50):
    """Return top-N recommendations using PCA-reduced content-based similarity (TF-IDF + PCA)."""
    sim = load_content_pca_similarity(_content_pca_sim_path, n_components=n_components)
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
from textblob import TextBlob

_sentiment_sim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../app/dataset/cosine_sim_sentiment.npy'))

def aggregate_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentiment polarity for each product by averaging sentiment of all its reviews."""
    def get_sentiment(text):
        if pd.isna(text) or not str(text).strip():
            return 0.0
        # If multiple reviews, split by comma and average
        reviews = str(text).split(',')
        scores = [TextBlob(r).sentiment.polarity for r in reviews if r.strip()]
        return np.mean(scores) if scores else 0.0
    agg = df[['product_id', 'review_content']].copy()
    agg['sentiment_score'] = agg['review_content'].apply(get_sentiment)
    return agg[['product_id', 'sentiment_score']]

def build_and_save_sentiment_similarity(df: pd.DataFrame, path: str = _sentiment_sim_path):
    """Compute and save cosine similarity matrix based on sentiment scores."""
    agg = aggregate_sentiment(df)
    X = agg[['sentiment_score']].values
    sim = cosine_similarity(X)
    save_similarity_to_file(sim, path)
    # Save index map (product_id -> row index)
    index_map = {str(pid): int(i) for i, pid in enumerate(agg['product_id'].astype(str).tolist())}
    save_index_map(index_map, _reviews_index_map_path)  # reuse index map path
    return sim

def load_sentiment_similarity(path: str = _sentiment_sim_path):
    if not os.path.exists(path):
        return build_and_save_sentiment_similarity(df, path)
    return load_similarity_from_file(path)

def get_sentiment_recommendations(product_id, N=5):
    """Return top-N recommendations using sentiment-based similarity."""
    sim = load_sentiment_similarity()
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
    agg = aggregate_sentiment(df)
    rec_ids = agg.iloc[product_indices]['product_id'].tolist()
    recommended_products = df[df['product_id'].isin(rec_ids)][['product_id', 'product_name', 'category', 'rating', 'discounted_price', 'img_link']]
    return recommended_products

# --- Topic Modeling-Based Recommendations (LDA on aggregated reviews) ---
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

_topic_sim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../app/dataset/cosine_sim_topic.npy'))

def build_and_save_topic_similarity(df: pd.DataFrame, path: str = _topic_sim_path, n_topics: int = 10):
    """Compute and save cosine similarity matrix based on LDA topic distributions of aggregated reviews."""
    agg = aggregate_reviews(df)
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

def load_topic_similarity(path: str = _topic_sim_path, n_topics: int = 10):
    if not os.path.exists(path):
        return build_and_save_topic_similarity(df, path, n_topics=n_topics)
    return load_similarity_from_file(path)

def get_topic_recommendations(product_id, N=5, n_topics: int = 10):
    """Return top-N recommendations using topic modeling (LDA) on reviews."""
    sim = load_topic_similarity(_topic_sim_path, n_topics=n_topics)
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

def get_reviewer_overlap_recommendations(product_id, N=5):
    """Return top-N products reviewed by users who also reviewed the given product."""
    global df
    users = set(df[df['product_id'] == product_id]['user_id'])
    if not users:
        return pd.DataFrame() # Fixed: return empty DataFrame instead of list
    
    overlap = df[df['user_id'].isin(users) & (df['product_id'] != product_id)]
    product_counts = overlap['product_id'].value_counts().head(N)
    
    if product_counts.empty:
        return pd.DataFrame() # Fixed: return empty DataFrame
        
    recs = df[df['product_id'].isin(product_counts.index)][['product_id', 'product_name', 'category', 'rating', 'discounted_price', 'img_link']].drop_duplicates()
    recs['overlap_count'] = recs['product_id'].map(product_counts)
    return recs.sort_values('overlap_count', ascending=False)

def get_hybrid_recommendations(product_id, N=5):
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
        get_recommendations_with_pca(product_id, N=n_each),
        get_review_recommendations(product_id, N=n_each),
        get_content_recommendations_pca(product_id, N=n_each),
        get_sentiment_recommendations(product_id, N=n_each),
        get_topic_recommendations(product_id, N=n_each),
        get_reviewer_overlap_recommendations(product_id, N=n_each),
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

def get_weighted_hybrid_recommendations(product_id, N=5, weights=None):
    """
    10-Model Hybrid Engine: Combines every strategy into one weighted score.
    Now includes SVD and KNN.
    """
    global df
    if df is None:
        return pd.DataFrame()

    model_keys = [
        'basic_cosine', 'pca_features', 'content_tfidf', 'content_pca', 
        'review_text', 'sentiment', 'topic_lda', 'reviewer_overlap', 
        'knn_numeric', 'svd_collaborative'
    ]

    if weights is None:
        weights = {k: 1.0/len(model_keys) for k in model_keys}

    # Normalize weights
    total = sum(weights.values())
    norm_weights = {k: v / total for k, v in weights.items()}

    # Prepare indices for KNN and SVD
    try:
        df_knn, X_knn = prepare_features_for_knn(df)
        p_idx = df[df['product_id'] == product_id].index[0]
    except Exception:
        p_idx = None

    n_pool = N * 2
    # Collect candidates from all 10 experts
    recs_to_combine = {
        'basic_cosine': get_recommendations(product_id, N=n_pool),
        'pca_features': get_recommendations_with_pca(product_id, N=n_pool),
        'content_tfidf': get_content_recommendations(product_id, N=n_pool),
        'content_pca': get_content_recommendations_pca(product_id, N=n_pool),
        'review_text': get_review_recommendations(product_id, N=n_pool),
        'sentiment': get_sentiment_recommendations(product_id, N=n_pool),
        'topic_lda': get_topic_recommendations(product_id, N=n_pool),
        'reviewer_overlap': get_reviewer_overlap_recommendations(product_id, N=n_pool)
    }
    
    # Add KNN and SVD if valid index exists
    if p_idx is not None:
        recs_to_combine['knn_numeric'] = get_knn_recommendations(df, X_knn, p_idx, n=n_pool)
        recs_to_combine['svd_collaborative'] = svd_product_recommendations(df, product_id, n=n_pool)

    # Weighted Borda count scoring
    from collections import defaultdict
    final_scores = defaultdict(float)
    
    for method, rec_data in recs_to_combine.items():
        # SAFE CHECK: Check if it's a non-empty DataFrame or a non-empty list
        is_empty = True
        if isinstance(rec_data, pd.DataFrame):
            is_empty = rec_data.empty
        elif isinstance(rec_data, list):
            is_empty = (len(rec_data) == 0)

        if not is_empty:
            w = norm_weights.get(method, 0)
            # Standardize to a list of product IDs for scoring
            pids = rec_data['product_id'].tolist() if isinstance(rec_data, pd.DataFrame) else [r['product_id'] for r in rec_data if isinstance(r, dict)]
            
            for rank, pid in enumerate(pids):
                # Points = weight * (pool_size - current_rank)
                final_scores[pid] += w * (len(pids) - rank)

    # Sort and return results
    sorted_pids = [pid for pid, _ in sorted(final_scores.items(), key=lambda x: -x[1]) if pid != product_id]
    top_pids = sorted_pids[:N]
    
    return df[df['product_id'].isin(top_pids)].copy().assign(
        hybrid_score=lambda d: d['product_id'].map(final_scores)
    ).sort_values('hybrid_score', ascending=False).reset_index(drop=True)

# --- ML-based Trending Products ---
def get_trending_products_ml(N=10, weights=None):
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




