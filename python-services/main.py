# Basic FastAPI ML Service Example
# Save this as python-services/main.py
from fastapi import FastAPI, Request, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Optional, Dict
from ml import print_head, df, get_recommendations, get_recommendations_with_pca, get_review_recommendations, get_content_recommendations, get_sentiment_recommendations, get_content_recommendations_pca, get_topic_recommendations, get_reviewer_overlap_recommendations, get_hybrid_recommendations, get_weighted_hybrid_recommendations

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

@app.get("/recommend_hybrid/{product_id}")
def recommend_hybrid(product_id: str, n: int = 5):
    """Return top-n hybrid recommendations for a given product_id (aggregated from all recommenders)."""
    recs = get_hybrid_recommendations(product_id, N=n)
    if isinstance(recs, list):
        return {"recommendations": recs}
    return {"recommendations": recs.to_dict(orient="records")}

@app.post("/recommend_weighted_hybrid/{product_id}")
def recommend_weighted_hybrid(product_id: str, n: int = 5, req: WeightedHybridRequest = Body(...)):
    """
    Return top-n weighted hybrid recommendations for a given product_id.
    Accepts weights as a JSON body: {"weights": {"pca": 0.2, "review": 0.1, ...}}
    """
    weights = req.weights
    recs = get_weighted_hybrid_recommendations(product_id, N=n, weights=weights)
    if isinstance(recs, list):
        return {"recommendations": recs}
    return {"recommendations": recs.to_dict(orient="records")}
