import pandas as pd
from sklearn.linear_model import LinearRegression
import os
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Get the absolute path to the CSV file
csv_path = os.path.join(os.path.dirname(__file__), '../app/dataset/amazon.csv')
product_features_path = os.path.join(os.path.dirname(__file__), '../app/dataset/product_features.csv')
abs_csv_path = os.path.abspath(csv_path)
abs_product_feature = os.path.abspath(product_features_path)
print(f"Resolved CSV path: {abs_csv_path}")
if not os.path.exists(abs_csv_path):
    raise FileNotFoundError(f"CSV file not found at: {abs_csv_path}")
df = pd.read_csv(abs_csv_path)
df_product_features = pd.read_csv(abs_product_feature)

from sklearn.metrics.pairwise import cosine_similarity

# Path to cached similarity matrix file
_sim_cache_path = os.path.join(os.path.dirname(__file__), '../app/dataset/cosine_sim.npy')
_abs_sim_cache_path = os.path.abspath(_sim_cache_path)

# Path to saved preprocessor
_encoder_path = os.path.join(os.path.dirname(__file__), '../app/dataset/preprocessor.joblib')
_abs_encoder_path = os.path.abspath(_encoder_path)

def save_similarity_to_file(matrix: np.ndarray, path: str = _abs_sim_cache_path) -> None:
    """Save similarity matrix to disk as a .npy file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, matrix)

def load_similarity_from_file(path: str = _abs_sim_cache_path) -> np.ndarray:
    """Load a similarity matrix saved with np.save/.npy. Raises FileNotFoundError if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Similarity file not found: {path}")
    return np.load(path, allow_pickle=False)

def load_or_build_similarity(path: str = _abs_sim_cache_path, feature_df: pd.DataFrame = None) -> np.ndarray:
    """Try to load the similarity matrix from `path`. If not present, compute from `feature_df`, save it, and return it."""
    try:
        return load_similarity_from_file(path)
    except FileNotFoundError:
        if feature_df is None:
            raise ValueError("feature_df must be provided to build the similarity matrix when cache is missing")
        from sklearn.metrics.pairwise import cosine_similarity
        mat = cosine_similarity(feature_df)
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
    if num_cols is None:
        remainder = 'passthrough'
    else:
        # we'll passthrough numeric columns explicitly; ColumnTransformer will preserve their order
        remainder = 'drop'

    # Create OneHotEncoder with compatibility for different sklearn versions
    try:
        # older sklearn versions accept `sparse`
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=sparse)
    except TypeError:
        # newer sklearn versions use `sparse_output`
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=sparse)

    preprocessor = ColumnTransformer(
        transformers=[('cat', ohe, cat_cols)],
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


def get_feature_matrix():
    """Return an encoded feature DataFrame for similarity computation.

    - If a saved preprocessor exists, apply it to `df_product_features`.
    - Otherwise infer categorical columns, build & save a preprocessor, then return the encoded DataFrame.
    """
    # Try to apply existing preprocessor
    if os.path.exists(_abs_encoder_path):
        try:
            encoded = apply_preprocessor(df_product_features, _abs_encoder_path)
            return encoded
        except Exception:
            # fall through to build if applying failed
            pass

    # Build a preprocessor from df_product_features
    cat_cols = infer_categorical_columns(df_product_features)
    # If no categorical columns found, still wrap with an empty transformer
    pre, encoded = build_and_save_preprocessor(df_product_features, cat_cols, num_cols=None, save_path=_abs_encoder_path, sparse=False)
    return encoded


def build_similarity_matrix():
    global cosine_sim_matrix
    if cosine_sim_matrix is None:
        # Get preprocessed feature matrix (DataFrame) and then build or load cached similarity
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

# You can add more ML functions or classes as needed

cosine_sim_matrix = None

def get_recommendations(product_id, N=5):
    # Get the index of the product that matches the product_id
    matrix = build_similarity_matrix()
    try:
        idx = df[df['product_id'] == product_id].index[0]
    except IndexError:
        print(f"Product ID '{product_id}' not found.")
        return []

    # Get the pairwise similarity scores of all products with that product
    sim_scores = list(enumerate(matrix[idx])) 

    # Sort the products based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the N most similar products (excluding itself)
    # The first element is the product itself, so skip it.
    sim_scores = sim_scores[1:N+1]

    # Get the product indices
    product_indices = [i[0] for i in sim_scores]

    # Return the top N most similar product details
    recommended_products = df.iloc[product_indices][['product_id', 'product_name', 'category', 'rating', 'discounted_price']]
    return recommended_products

print("Recommendation function 'get_recommendations' updated to use reduced feature set.")



