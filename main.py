from fastapi import FastAPI
import torch
import numpy as np
import joblib

app = FastAPI()

# Load trained TF-IDF vectorizer and Pytorch model
tfidf = joblib.load()
device = torch.device('cuda' if torch.is_available() else 'cpu')
model = torch.load('', map_location=device)
model.eval()


app.get("/")
def home_page():
    return f"welcome to the mental health hub homepage"


@app.post("/predict")
def predict_mental_health(payload: TextPayload):
    # 1 Preprocess and vectorize the input text
    vectorized_text = tfidf.transform([payload.text]).toarray()
    tensor_input = torch.tensor(vectorized_text, dtype=torch.float32).unsqueeze(1).to(device)

    # model prediction
    with torch.no_grad():
        outputs = model(tensor_input)
        logits = outputs[0]
        probability = torch.softmax(logits, dim=1).cpu().numpy()[0][1]
        prediction = torch.argmax(logits, dim=1).cpu().item()


    return {
        "prediction": prediction,
        "probability_socre": float(probability),
        "status": "Flagged" if prediction == 1 else "Normal"
    }