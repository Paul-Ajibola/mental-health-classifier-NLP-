import os

import requests
import streamlit as st


st.set_page_config(page_title="Mental Health Text Classifier", page_icon="🧠", layout="centered")
backend_url = os.environ.get("BACKEND_URL", "https://mental-health-classifier-nlp-1.onrender.com").rstrip("/")

st.title("🧠 Mental Health Text Classifier")
st.warning(
    "Research demonstration only. This classifier cannot diagnose a condition, "
    "assess immediate danger, or replace help from a qualified professional. "
    "If you may be in immediate danger, contact local emergency services or a trusted person."
)
st.write("Enter text to see the model's research classification. Avoid entering identifying information.")

user_text = st.text_area(
    "Text to analyze",
    placeholder="Type or paste text here (up to 10,000 characters)...",
    height=150,
    max_chars=10_000,
)

if st.button("Analyze text", type="primary"):
    if not user_text.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        try:
            response = requests.post(
                f"{backend_url}/predict",
                json={"text": user_text},
                timeout=15,
            )
            if response.ok:
                result = response.json()
                prediction = result["prediction"]
                probability = result["flagged_probability"]
                status = result["status"]

                st.success("Analysis complete")
                col1, col2 = st.columns(2)
                col1.metric("Model classification", status)
                col2.metric("Flagged-class probability", f"{probability:.1%}")
                st.caption(
                    f"Class {prediction}. The probability is a model score, not a clinical risk estimate."
                )
            else:
                try:
                    detail = response.json().get("detail", response.text)
                except ValueError:
                    detail = response.text
                st.error(f"Backend returned {response.status_code}: {detail}")
        except requests.exceptions.Timeout:
            st.error("The backend did not respond in time. Please try again.")
        except requests.exceptions.ConnectionError:
            st.error(
                f"Could not connect to {backend_url}. Start the API and check BACKEND_URL."
            )
        except (KeyError, ValueError, requests.exceptions.RequestException) as exc:
            st.error(f"Could not process the backend response: {exc}")

st.divider()
st.caption("NLP Project (Paul Ajibola).")
