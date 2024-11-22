import streamlit as st
import pandas as pd
from prediction import predict

# Upload CSV
st.title("Student Dropout Prediction Dashboard")
uploaded_file = st.file_uploader("Upload your dataset", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(data)
    
    # Prediction
    try:
        predictions = predict(data)
        data["Predictions"] = predictions
        st.write("Predicted Results:")
        st.dataframe(data)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
