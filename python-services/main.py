# Basic FastAPI ML Service Example
# Save this as python-services/main.py
from fastapi import FastAPI, Request, Query, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Optional, Dict
from .ml import print_head, df, get_recommendations, get_recommendations_with_pca, get_review_recommendations, get_content_recommendations, get_sentiment_recommendations, get_content_recommendations_pca, get_topic_recommendations, get_reviewer_overlap_recommendations, get_weighted_hybrid_recommendations, get_trending_products_ml, svd_product_recommendations, prepare_features_for_knn, get_knn_recommendations
from pymongo import MongoClient
import os
import pandas as pd

app = FastAPI()

# Allow CORS for all origins (for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    features: list[float]

class WeightedHybridRequest(BaseModel):
    weights: Optional[Dict[str, float]] = None

@app.get("/")
def root():
    return {"message": "ML Service is running!"}

@app.post("/predict")
def predict(request: PredictRequest) -> Any:
    # Dummy prediction logic, replace with your ML model
    prediction = sum(request.features)
    return {"prediction": prediction}

@app.get("/head")
def get_head():
    print_head()  # This will print to the server console
    # Optionally, return the head as JSON
    import pandas as pd
    return {"head": df.head().to_dict(orient="records")}

@app.get("/recommend/{product_id}")
def recommend(product_id: str, n: int = 5):
    """Return top-n recommendations for a given product_id."""
    recs = get_recommendations(product_id, N=n)
    # If get_recommendations returns a list (e.g. when product not found), return it directly
    if isinstance(recs, list):
        return {"recommendations": recs}
    # Otherwise it's a DataFrame — convert to list of dicts for JSON
    return {"recommendations": recs.to_dict(orient="records")}

@app.get("/recommend_pca/{product_id}")
def recommend_pca(product_id: str, n: int = 5, n_components: int = 50):
    """Return top-n PCA-based recommendations for a given product_id."""
    recs = get_recommendations_with_pca(product_id, N=n, n_components=n_components)
    if isinstance(recs, list):
        return {"recommendations": recs}
    return {"recommendations": recs.to_dict(orient="records")}

@app.get("/recommend_reviews/{product_id}")
def recommend_reviews(product_id: str, n: int = 5):
    """Return top-n review-based recommendations for a given product_id."""
    recs = get_review_recommendations(product_id, N=n)
    if isinstance(recs, list):
        return {"recommendations": recs}
    return {"recommendations": recs.to_dict(orient="records")}

@app.get("/recommend_content/{product_id}")
def recommend_content(product_id: str, n: int = 5):
    """Return top-n content-based recommendations for a given product_id (using all text fields)."""
    recs = get_content_recommendations(product_id, N=n)
    if isinstance(recs, list):
        return {"recommendations": recs}
    return {"recommendations": recs.to_dict(orient="records")}

@app.get("/recommend_sentiment/{product_id}")
def recommend_sentiment(product_id: str, n: int = 5):
    """Return top-n sentiment-based recommendations for a given product_id."""
    recs = get_sentiment_recommendations(product_id, N=n)
    if isinstance(recs, list):
        return {"recommendations": recs}
    return {"recommendations": recs.to_dict(orient="records")}

@app.get("/recommend_content_pca/{product_id}")
def recommend_content_pca(product_id: str, n: int = 5, n_components: int = 50):
    """Return top-n PCA-reduced content-based recommendations for a given product_id (TF-IDF + PCA)."""
    recs = get_content_recommendations_pca(product_id, N=n, n_components=n_components)
    if isinstance(recs, list):
        return {"recommendations": recs}
    return {"recommendations": recs.to_dict(orient="records")}

@app.get("/recommend_topic/{product_id}")
def recommend_topic(product_id: str, n: int = 5, n_topics: int = 10):
    """Return top-n topic modeling (LDA) recommendations for a given product_id."""
    recs = get_topic_recommendations(product_id, N=n, n_topics=n_topics)
    if isinstance(recs, list):
        return {"recommendations": recs}
    return {"recommendations": recs.to_dict(orient="records")}

@app.get("/recommend_reviewer_overlap/{product_id}")
def recommend_reviewer_overlap(product_id: str, n: int = 5):
    """Return top-n reviewer-overlap recommendations for a given product_id."""
    recs = get_reviewer_overlap_recommendations(product_id, N=n)
    if isinstance(recs, list):
        return {"recommendations": recs}
    return {"recommendations": recs.to_dict(orient="records")}

@app.get("/recommend_svd/{product_id}")
def recommend_svd(product_id: str, n: int = 5):
    """Return top-n SVD recommendations for a given product_id."""
    recs = svd_product_recommendations(df, product_id, n=n)
    if isinstance(recs, list):
        return {"recommendations": recs}
    return {"recommendations": recs.to_dict(orient="records")}

@app.get("/recommend_knn/{product_id}")
def recommend_knn(product_id: str, n: int = 5):
    """Return top-n KNN-based recommendations for a given product_id."""
    try:
        df_knn, X_knn = prepare_features_for_knn(df)
        product_indices = df_knn[df_knn['product_id'] == product_id].index
        if product_indices.empty:
            return {"recommendations": []}
        product_idx = product_indices[0]
        recs = get_knn_recommendations(df_knn, X_knn, product_idx, n=n)
        if isinstance(recs, list):
            return {"recommendations": recs}
        return {"recommendations": recs.to_dict(orient="records")}
    except Exception as e:
        print(f"Error in KNN recommendation: {e}")
        return {"recommendations": []}

@app.post("/recommend_weighted_hybrid/{product_id}")
def recommend_weighted_hybrid(
    product_id: str,
    n: int = 5,
    req: WeightedHybridRequest = Body(...)
):
    weights = req.weights
    if not weights:
        # Default weights favoring your high-performance models (SVD, KNN, Overlap)
        weights = {
            'basic_cosine': 0.0293, 
            'pca_features': 0.0883,
            'content_tfidf': 0.2967,
            'content_pca': 0.1711, 
            'review_text': 0.0779, 
            'sentiment': 0.0935,
            'topic_lda': 0.0709, 
            'reviewer_overlap': 0.1381, 
            'knn_numeric': 0.0085,
            'svd_collaborative': 0.0257
        }

    recs = get_weighted_hybrid_recommendations(product_id, N=n, weights=weights)

    if isinstance(recs, pd.DataFrame):
        return {"recommendations": recs.to_dict(orient="records")}
    return {"recommendations": []}

@app.get("/trending_products")
def trending_products(n: int = 8, min_rating: float = 4.0, min_reviews: int = 1):
    """
    Return top-n trending products based on the number of positive reviews (rating >= min_rating).
    """
    # MongoDB connection settings (reuse env vars if available)
    MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://raquelbastian_db_user:oXu3M164d7HEwcVL@capstone.jucteam.mongodb.net")
    MONGO_DB = os.environ.get("MONGO_DB", "capstone")
    MONGO_COLLECTION = os.environ.get("MONGO_COLLECTION", "products")
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    collection = db[MONGO_COLLECTION]

    # Aggregate products by counting positive reviews (rating >= min_rating)
    pipeline = [
        {"$unwind": "$reviews"},
        {"$match": {"reviews.rating": {"$gte": min_rating}}},
        {"$group": {
            "_id": "$product_id",
            "product_name": {"$first": "$product_name"},
            "category": {"$first": "$category"},
            "img_link": {"$first": "$img_link"},
            "discounted_price": {"$first": "$discounted_price"},
            "rating": {"$first": "$rating"},
            "positive_review_count": {"$sum": 1}
        }},
        {"$match": {"positive_review_count": {"$gte": min_reviews}}},
        {"$sort": {"positive_review_count": -1, "rating": -1}},
        {"$limit": n}
    ]
    results = list(collection.aggregate(pipeline))
    # Convert _id to product_id for frontend
    for r in results:
        r["product_id"] = r.pop("_id")
    return {"trending": results}

@app.get("/trending_products_ml")
def trending_products_ml(n: int = 8):
    """
    Return top-n trending products using ML-based hybrid score (positive reviews, avg rating, sentiment).
    """
    trending = get_trending_products_ml(N=n)
    # Convert DataFrame to list of dicts for JSON
    return {"trending": trending.to_dict(orient="records")}
