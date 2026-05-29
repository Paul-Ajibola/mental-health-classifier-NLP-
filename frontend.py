import streamlit as st
import requests

# Set page configuration
st.set_page_config(
    page_title="Mental Health Hub",
    page_icon="🧠",
    layout="centered"
)

# Define the FastAPI backend URL
# Update this URL if your backend is hosted elsewhere or on a different port
BACKEND_URL = "http://127.0.0.1:8000/predict"

# App Header
st.title("🧠 Mental Health Hub Analysis")
st.write(
    "Enter the text below to analyze the emotional tone or potential flags. "
    "The text will be processed by our trained PyTorch model via the FastAPI backend."
)

st.divider()

# Text Input Area
user_text = st.text_area(
    "Input Text for Analysis:",
    placeholder="Type or paste the text content here...",
    height=150
)

# Prediction Logic
if st.button("Analyze Text", type="primary"):
    if not user_text.strip():
        st.warning("Please enter some text before clicking analyze.")
    else:
        with st.spinner("Sending data to backend and running inference..."):
            try:
                # Prepare payload exactly matching the TextPayload Pydantic model
                payload = {"text": user_text}
                
                # Make POST request to FastAPI
                response = requests.post(BACKEND_URL, json=payload)
                
                if response.status_color == 200:
                    result = response.json()
                    
                    # Extract fields from the API response
                    prediction = result.get("prediction")
                    probability = result.get("probability_socre")  # Matching your typo
                    status = result.get("status")
                    
                    st.success("Analysis Complete!")
                    
                    # Display Results using Streamlit Metrics and Alert Boxes
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label="Prediction Class", value=str(prediction))
                    with col2:
                        st.metric(label="Probability Score", value=f"{probability:.4f}")
                    
                    # Visual status alert based on model output
                    if status == "Flagged":
                        st.error(f"🚨 Status: **{status}**")
                    else:
                        st.sidebar.success(f"✅ Status: **{status}**")
                        st.info(f"🌿 Status: **{status}**")
                        
                else:
                    st.error(f"Backend Error: Received status code {response.status_code}")
                    st.json(response.text)
                    
            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to the FastAPI backend. "
                    "Make sure your FastAPI server is running on `http://127.0.0.1:8000`"
                )
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

# Footer info
st.divider()
st.caption("Note: This tool is driven by an AI classification model for research and analysis purposes.")