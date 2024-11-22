import pandas as pd

def preprocess_data(input_data, encoders):
    """
    Preprocess input data for prediction.
    :param input_data: Pandas DataFrame containing input data.
    :param encoders: Dictionary of LabelEncoders.
    :return: Processed DataFrame.
    """
    # Ensure input is a DataFrame
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    
    # Encode categorical features
    for col, encoder in encoders.items():
        if col in input_data.columns:
            input_data[col] = encoder.transform(input_data[col])
    
    return input_data
