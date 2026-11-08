from pathlib import Path
import os

import joblib
from fastapi import FastAPI, HTTPException

from schema import PredictionResponse, TextPayload


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "models" / "text_classifier.joblib"))
MAX_TEXT_LENGTH = 10_000


app = FastAPI(
    title="Mental Health Text Classifier",
    description="Research demonstration only; not a diagnostic or crisis-response tool.",
    version="1.0.0",
)


@app.on_event("startup")
def load_model() -> None:
    """Load the trained sklearn pipeline once when the API starts."""
    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Run `python train_model.py` from the project directory first."
        )
    app.state.classifier = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": hasattr(app.state, "classifier")}


@app.get("/")
def home_page():
    return {
        "name": app.title,
        "docs": "/docs",
        "notice": "Research demonstration only; predictions are not clinical advice.",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_mental_health(payload: TextPayload):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text must not be empty or whitespace.")
    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"Text must be no longer than {MAX_TEXT_LENGTH} characters.",
        )

    classifier = getattr(app.state, "classifier", None)
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    prediction = int(classifier.predict([text])[0])
    probabilities = classifier.predict_proba([text])[0]
    positive_index = list(classifier.classes_).index(1)
    probability = float(probabilities[positive_index])
    return PredictionResponse(
        prediction=prediction,
        flagged_probability=probability,
        status="Flagged" if prediction == 1 else "Not flagged",
    )
