import streamlit as st
import pandas as pd
from prediction import load_model, predict_status

# Load model and encoders
model, encoders = load_model()

# Streamlit UI
st.title("Student Dropout Prediction Dashboard")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file
    data = pd.read_csv(uploaded_file, delimiter=';')
    st.write("Uploaded Data:", data.head())

    # Preprocess and predict
    try:
        predictions = predict_status(data, model, encoders)
        data['Predicted_Status'] = predictions
        st.success("Prediction Successful!")
        st.write(data)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
