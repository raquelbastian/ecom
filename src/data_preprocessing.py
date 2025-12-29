"""
python-services/clean_data.py

Clean the raw CSV (app/dataset/amazon.csv) into two parquet artifacts:
 - amazon_clean.parquet   (full cleaned dataset)
 - product_features.parquet (slim dataset with features used by ML)

Also emits index_map.json mapping product_id -> row index for the feature table.

Usage:
  python3 clean_data.py --csv ../app/dataset/amazon.csv --outdir ../app/dataset --chunksize 10000

This script is careful about memory by reading in chunks but will concatenate cleaned chunks before writing parquet.
It performs:
 - whitespace trimming and basic normalization for string columns
 - numeric parsing for rating/price columns
 - drop rows missing product_id or product_name
 - deduplicate by product_id (keeps first occurrence)
 - writes parquet artifacts and index map

"""
from pathlib import Path
import argparse
import re
import json
import sys
import pandas as pd
import numpy as np

# default dataset directory (relative to this script)
_DATA_DIR = Path(__file__).parent.joinpath('../data/processed').resolve()
_RAW_DATA_DIR = Path(__file__).parent.joinpath('../data/raw').resolve()

# default filenames
_DEFAULT_CSV = _RAW_DATA_DIR.joinpath('amazon.csv')
_AMAZON_CLEAN = _DATA_DIR.joinpath('amazon_clean.parquet')
_PRODUCT_FEATURES = _DATA_DIR.joinpath('product_features.parquet')
_INDEX_MAP = _DATA_DIR.joinpath('index_map.json')

# columns we prefer to keep in the product_features artifact (if present)
_PREFERRED_FEATURE_COLS = [
    'product_id', 'product_name', 'category', 'rating', 'price', 'discounted_price', 'img_link'
]

# regex to extract numeric values (digits, dot, minus)
_RE_NUM = re.compile(r'[-+]?[0-9]*\.?[0-9]+')


def clean_text(val: object) -> object:
    """Normalize text values: strip whitespace, unify spaces, remove control chars."""
    if pd.isna(val):
        return val
    try:
        s = str(val)
    except Exception:
        return val
    # remove control characters
    s = re.sub(r"[\x00-\x1f\x7f]+", " ", s)
    # normalize spaces
    s = re.sub(r"\s+", " ", s)
    # strip
    s = s.strip()
    return s


def parse_price(val: object) -> float:
    """Try to parse a price-like string into a float. Returns NaN on failure."""
    if pd.isna(val):
        return np.nan
    s = str(val)
    # common replacements
    s = s.replace('\u00a0', ' ')
    # find first numeric token
    m = _RE_NUM.search(s.replace(',', ''))
    if not m:
        return np.nan
    try:
        return float(m.group(0))
    except Exception:
        return np.nan


def parse_rating(val: object) -> float:
    """Parse rating to float (e.g. '4.5 out of 5' -> 4.5)."""
    if pd.isna(val):
        return np.nan
    s = str(val)
    m = _RE_NUM.search(s)
    if not m:
        return np.nan
    try:
        return float(m.group(0))
    except Exception:
        return np.nan


def clean_img_link(val: object) -> object:
    """Clean and validate image link URLs. Returns NaN if invalid."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if not s or not (s.startswith('http://') or s.startswith('https://')):
        return np.nan
    # Optionally filter out known bad patterns (e.g., placeholder images)
    if 'no_image' in s or 'placeholder' in s:
        return np.nan
    return s


def build_index_map(df: pd.DataFrame, id_col: str = 'product_id') -> dict:
    """Return dict mapping product_id (string) -> integer index into df."""
    if id_col not in df.columns:
        raise KeyError(f"id column '{id_col}' not found in DataFrame")
    idx_map = {str(pid): int(i) for i, pid in enumerate(df[id_col].astype(str).tolist())}
    return idx_map


def clean_chunk(df_chunk: pd.DataFrame) -> pd.DataFrame:
    # normalize string columns
    obj_cols = df_chunk.select_dtypes(include=['object', 'string']).columns.tolist()
    for c in obj_cols:
        df_chunk[c] = df_chunk[c].apply(clean_text)

    # parse rating if present
    if 'rating' in df_chunk.columns:
        df_chunk['rating'] = df_chunk['rating'].apply(parse_rating)

    # parse price-like columns
    for col in ['actual_price', 'discounted_price']:
        if col in df_chunk.columns:
            df_chunk[col] = df_chunk[col].apply(parse_price)

    # clean img_link column if present
    if 'img_link' in df_chunk.columns:
        df_chunk['img_link'] = df_chunk['img_link'].apply(clean_img_link)

    # clean price columns
    for col in ['actual_price', 'discounted_price']:
        if col in df_chunk.columns:
            df_chunk[col] = df_chunk[col].apply(clean_currency)

    return df_chunk



def clean_currency(val):
    # This regex finds the first sequence of digits and decimal points
    _re_num = re.compile(r'[-+]?[0-9]*\.?[0-9]+')
    if pd.isna(val) or val == '':
        return 0.0 # Return 0.0 for empty or missing values
    
    # Ensure it's a string and remove commas
    s = str(val).replace(',', '')
    
    # Find the numeric part
    match = _re_num.search(s)
    
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return 0.0
    return 0.0


def main(csv_path: Path | str = None, out_dir: Path | str = None, chunksize: int = 10000):
    csv_path = Path(csv_path) if csv_path is not None else _DEFAULT_CSV
    out_dir = Path(out_dir) if out_dir is not None else _DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        sys.exit(2)

    cleaned_chunks = []
    total_rows = 0
    print(f"Reading and cleaning CSV in chunks (chunksize={chunksize}): {csv_path}")
    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize, dtype=str, keep_default_na=False)):
        print(f"  processing chunk {i} (rows={len(chunk)})")
        # treat empty strings as NaN for cleaning
        chunk = chunk.replace({'': pd.NA})
        chunk = clean_chunk(chunk)
        cleaned_chunks.append(chunk)
        total_rows += len(chunk)

    if len(cleaned_chunks) == 0:
        print("No data read from CSV.")
        sys.exit(1)

    df_clean = pd.concat(cleaned_chunks, ignore_index=True)
    print(f"Concatenated {len(cleaned_chunks)} chunks -> {len(df_clean)} rows")

    # ensure product_id and product_name exist
    if 'product_id' not in df_clean.columns or 'product_name' not in df_clean.columns:
        print("Required columns 'product_id' or 'product_name' are missing from the CSV.")
        sys.exit(2)

    # drop rows missing id or name
    before = len(df_clean)
    df_clean = df_clean.dropna(subset=['product_id', 'product_name'])
    after = len(df_clean)
    print(f"Dropped {before - after} rows missing product_id or product_name")

    # convert rating/price to numeric types where possible
    if 'rating' in df_clean.columns:
        df_clean['rating'] = pd.to_numeric(df_clean['rating'], errors='coerce')
    for c in ['price', 'discounted_price']:
        if c in df_clean.columns:
            df_clean[c] = pd.to_numeric(df_clean[c], errors='coerce')

    # deduplicate by product_id, keep first
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['product_id'], keep='first').reset_index(drop=True)
    after = len(df_clean)
    print(f"Dropped {before - after} duplicate product_id rows")

    # create product_features slim table by selecting preferred columns if available
    present_pref_cols = [c for c in _PREFERRED_FEATURE_COLS if c in df_clean.columns]
    if len(present_pref_cols) == 0:
        # fallback: use all columns
        df_features = df_clean.copy()
        print("No preferred feature columns found; using full cleaned table as product_features")
    else:
        df_features = df_clean[present_pref_cols].copy()
        print(f"product_features will contain columns: {present_pref_cols}")

    # reset index and ensure deterministic ordering (sort by product_id string)
    try:
        df_features['product_id'] = df_features['product_id'].astype(str)
        df_features = df_features.sort_values(by='product_id').reset_index(drop=True)
    except Exception:
        df_features = df_features.reset_index(drop=True)

    # save artifacts
    amazon_clean_path = out_dir.joinpath('amazon_clean.parquet')
    product_features_path = out_dir.joinpath('product_features.parquet')
    index_map_path = out_dir.joinpath('index_map.json')

    print(f"Writing cleaned full table to: {amazon_clean_path}")
    df_clean.to_parquet(amazon_clean_path, index=False)

    print(f"Writing product features table to: {product_features_path}")
    df_features.to_parquet(product_features_path, index=False)

    # build and save index map
    try:
        idx_map = build_index_map(df_features, id_col='product_id')
        with open(index_map_path, 'w', encoding='utf-8') as fh:
            json.dump(idx_map, fh, ensure_ascii=False, indent=2)
        print(f"Wrote index map to: {index_map_path} (entries={len(idx_map)})")
    except Exception as e:
        print(f"Failed to write index map: {e}")

    print("Clean complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean amazon CSV and produce parquet artifacts for ML')
    parser.add_argument('--csv', type=str, default=str(_DEFAULT_CSV), help='Path to raw amazon CSV')
    parser.add_argument('--outdir', type=str, default=str(_DATA_DIR), help='Output directory for parquet artifacts')
    parser.add_argument('--chunksize', type=int, default=10000, help='CSV reader chunksize')
    args = parser.parse_args()
    main(csv_path=args.csv, out_dir=args.outdir, chunksize=args.chunksize)
