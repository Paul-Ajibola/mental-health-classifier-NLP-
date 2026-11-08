# Mental Health Text Classifier

Research demonstration that classifies text into dataset labels `0` and `1`. It is not a diagnostic, clinical risk assessment, or crisis-response service.

## Run locally

Use Python 3.10 or newer. From this directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
```

Start the API in one terminal:

```powershell
uvicorn main:app --host 127.0.0.1 --port 8000
```

Start the Streamlit interface in another terminal:

```powershell
streamlit run frontend.py
```

The API exposes `/health`, `/predict`, and interactive documentation at `/docs`. Set `BACKEND_URL` if the API is not at `http://127.0.0.1:8000`. Set `MODEL_PATH` to use a model artifact stored elsewhere.

Training uses a fixed stratified 80/20 split and saves the fitted TF-IDF and Multinomial Naive Bayes pipeline to `models/text_classifier.joblib`. The API requires this artifact; run the training command before starting it. Only load model artifacts from a trusted source because joblib files can execute code when loaded.

## Data and privacy

Do not enter identifying or sensitive personal information. This app does not intentionally persist submitted text, but the browser, network, hosting platform, or server logs may retain request data. Configure those systems accordingly before exposing the app to other users. A model score is not a clinical probability, and a `Not flagged` result does not mean someone is safe.
