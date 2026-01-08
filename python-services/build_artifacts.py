import pandas as pd
import joblib
import json
import os
import sys
from pymongo import MongoClient
import ml # Ensure ml.py is in the same directory
import numpy as np

def check_mongo_connection():
    """Verifies MongoDB connectivity before starting the build."""
    try:
        client = MongoClient(ml.MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("MongoDB Connection: SUCCESS")
        return client
    except Exception as e:
        print(f"MongoDB Connection: FAILED - {e}")
        sys.exit(1)

def load_and_clean_data(client):
    """Extracts, deduplicates, and cleans raw data from MongoDB."""
    db = client[ml.MONGO_DB]
    collection = db[ml.MONGO_COLLECTION]
    
    products = list(collection.find())
    if not products:
        raise ValueError("No products found in MongoDB collection.")
    
    for p in products:
        p.pop('_id', None)
    df = pd.DataFrame(products)

    # --- DEDUPLICATION ---
    initial_count = len(df)
    df = df.drop_duplicates(subset=['product_id'], keep='last')
    print(f"Deduplication: Removed {initial_count - len(df)} duplicate products.")

    # --- CLEANING ---
    for col in ['discounted_price', 'actual_price']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('₹', '').str.replace(',', '')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    if 'discount_percentage' in df.columns:
        df['discount_percentage'] = df['discount_percentage'].astype(str).str.replace('%', '')
        df['discount_percentage'] = pd.to_numeric(df['discount_percentage'], errors='coerce').fillna(0)
    
    if 'rating' in df.columns:
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)

    return df

def run_automated_build(output_dir='../app/dataset'):
    os.makedirs(output_dir, exist_ok=True)
    client = check_mongo_connection()
    df = load_and_clean_data(client)


    # DELETE OLD CACHES to prevent the 'Hero-to-Zero' bug
    paths_to_clear = [
        os.path.join(output_dir, 'cosine_sim_sentiment.npy'),
        os.path.join(output_dir, 'sentiment_scores.parquet')
    ]
    for p in paths_to_clear:
        if os.path.exists(p):
            os.remove(p)
            print(f"Cleared old cache: {p}")

    # Now run the generation
    ml.build_and_save_sentiment_similarity(df)
    
    # 1. Save Parquet
    parquet_path = os.path.join(output_dir, 'product_features.parquet')
    df.to_parquet(parquet_path, index=False)
    print(f"Artifact 1/4: Saved Parquet to {parquet_path}")

    # Also save a copy to data/processed
    processed_dir = '../data/processed'
    os.makedirs(processed_dir, exist_ok=True)
    processed_parquet_path = os.path.join(processed_dir, 'product_features.parquet')
    df.to_parquet(processed_parquet_path, index=False)
    print(f"Saved additional copy to {processed_parquet_path}")

    # 2. Generate Index Map
    index_map = {str(pid): i for i, pid in enumerate(df['product_id'])}
    index_map_path = os.path.join(output_dir, 'index_map.json')
    with open(index_map_path, 'w') as f:
        json.dump(index_map, f)
    print(f"Artifact 2/4: Index Map generated")

    print("Generating and Validating Similarity Matrices...")
    
    # Matrix Generation Tasks
    tasks = [
        ('cosine_sim.npy', ml.build_similarity_matrix),
        ('cosine_sim_pca.npy', ml.build_similarity_matrix_with_pca),
        ('cosine_sim_content.npy', lambda: ml.build_and_save_content_similarity(df)),
        ('cosine_sim_content_pca.npy', lambda: ml.build_and_save_content_pca_similarity(df)), 
        ('cosine_sim_reviews.npy', lambda: ml.build_and_save_review_similarity(df)),
        ('cosine_sim_sentiment.npy', lambda: ml.build_and_save_sentiment_similarity(df)),
        ('cosine_sim_topic.npy', lambda: ml.build_and_save_topic_similarity(df)),
        ('cosine_sim_knn.npy', lambda: ml.build_and_save_knn_numeric_similarity(df))
    ]

    for filename, func in tasks:
        print(f"Processing {filename}...")
        func()  # Trigger the build in ml.py
        
        # Immediate Sanity Check
        path = os.path.join(output_dir, filename)
        try:
            validate_matrix(path, filename)
        except Exception as e:
            print(f"🛑 BUILD HALTED: {e}")
            sys.exit(1)

    print("🚀 All artifacts are healthy and synchronized.")
    
   
    if not df.empty:
        ml.svd_product_recommendations(df, df['product_id'].iloc[0]) # SVD Factors

    # 4. Save Joblib Models
    cat_cols = ['category', 'product_name']
    num_cols = ['rating', 'discounted_price', 'actual_price', 'discount_percentage']
    
    pre_path = os.path.join(output_dir, 'preprocessor.joblib')
    ml.build_and_save_preprocessor(df, cat_cols, num_cols=num_cols, save_path=pre_path)
    
    print("Build Complete: All artifacts generated successfully.")

def validate_matrix(matrix_path, name):
    """
    Sanity Check: Ensures the generated matrix is mathematically valid.
    Prevents 'Hero-to-Zero' performance drops.
    """
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"CRITICAL: {name} was not created!")
    
    # Load with mmap to save RAM during validation
    mat = np.load(matrix_path, mmap_mode='r')
    
    # Check 1: Is it empty or all zeros?
    if np.all(mat == 0):
        raise ValueError(f"CRITICAL: {name} is a zero-matrix. Data signal lost!")
    
    # Check 2: Check for NaNs
    if np.isnan(mat).any():
        print(f"WARNING: {name} contains NaNs. Filling with zeros...")
    
    # Check 3: Identity Check (Diagonal should be ~1.0 for cosine similarity)
    diag_avg = np.diag(mat).mean()
    if diag_avg < 0.9:
        print(f"WARNING: {name} has a weak diagonal ({diag_avg:.2f}). Check indexing.")

    print(f"✅ {name} validated successfully.")

if __name__ == "__main__":
    run_automated_build()