import sys
import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from preprocess import clean_sec_text
from utils import load_artifact

app = FastAPI(title="SEC 10-K API")

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'best_model.joblib')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), '..', 'tfidf_vectorizer.joblib')

production_model = None
production_vectorizer = None
MODEL_FILENAME = "best_model.joblib"
VECTORIZER_FILENAME = "tfidf_vectorizer.joblib"

@app.on_event("startup")
def load_models():
    global production_model, production_vectorizer
    production_model = load_artifact(MODEL_FILENAME)
    production_vectorizer = load_artifact(VECTORIZER_FILENAME)

class PredictRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_document(payload: PredictRequest):
    if not production_model or not production_vectorizer:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded.")
        
    cleaned = clean_sec_text(payload.text)
    features = production_vectorizer.transform([cleaned]).toarray()

    prediction = production_model.predict(features)
    label = prediction[0]
    if hasattr(label, "__len__") and not isinstance(label, (str, int)):
        label = label[0]

    probabilities = production_model.predict_proba(features)[0]
    confidence = np.max(probabilities)

    return {
        "label": int(label),
        "confidence": float(round(confidence, 4))
    }

