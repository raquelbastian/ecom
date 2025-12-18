# Basic FastAPI ML Service Example
# Save this as python-services/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any
from ml import print_head, df, get_recommendations

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
