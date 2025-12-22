"""Build artifacts for the ML service.

This script will:
- Ensure product features are available (Parquet preferred; will write parquet if only CSV exists).
- Build or load the preprocessor and encoded features (saved to encoded_features.parquet).
- Build or load the cosine similarity matrix (saved to cosine_sim.npy).
- Build and save an index_map.json mapping product_id -> row index.

Run this script from the repository root using:
  cd python-services && python3 build_artifacts.py

"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
print("Running build_artifacts from:", ROOT)

# Ensure python path includes this directory so `ml` can be imported
sys.path.insert(0, str(ROOT))

try:
    import ml
except Exception as e:
    print("Failed to import ml module:", e)
    raise

# 1) Load product features (this will create parquet if needed)
print("Loading product features...")
df_pf = ml.load_product_features()
print(f"Product features shape: {df_pf.shape}")

# 2) Build or load preprocessor & encoded features via get_feature_matrix()
print("Building/applying preprocessor and encoded features...")
encoded = ml.get_feature_matrix()
print(f"Encoded features shape: {encoded.shape}")

# 3) Build or load similarity matrix
print("Building/loading similarity matrix (may take time)...")
mat = ml.build_similarity_matrix()
print("Similarity matrix shape:", getattr(mat, 'shape', 'unknown'))

# 4) Confirm index map exists or create it
try:
    idx_map = ml.load_index_map()
    print(f"Loaded index map with {len(idx_map)} entries")
except Exception:
    print("Index map missing; creating from product_features...")
    try:
        df_pf = ml.load_product_features()
        index_map = {str(pid): int(i) for i, pid in enumerate(df_pf['product_id'].astype(str).tolist())}
        ml.save_index_map(index_map)
        print(f"Index map saved with {len(index_map)} entries")
    except Exception as e:
        print("Failed to create index map:", e)

print("Artifact build complete.")
