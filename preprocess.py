# preprocess.py
import pandas as pd

def preprocess_data(data, encoders):
    """
    Preprocess the input data for prediction.
    Args:
        data (pd.DataFrame): Raw input data.
        encoders (dict): Dictionary of encoders for categorical features.
    Returns:
        pd.DataFrame: Processed data ready for prediction.
    """
    # Handle missing values
    data.fillna(0, inplace=True)
    
    # Encode categorical columns
    for col, encoder in encoders.items():
        if col in data.columns:
            data[col] = encoder.transform(data[col].fillna("Unknown"))

    # Ensure numerical columns are of the right type
    numerical_cols = data.select_dtypes(include='number').columns
    data[numerical_cols] = data[numerical_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    return data
