import pickle

def load_model():
    """
    Load the trained model and encoders from file.
    :return: Tuple (model, encoders)
    """
    with open("model.pkl", "rb") as file:
        data = pickle.load(file)
    return data["model"], data["encoders"]

def predict_status(input_data, model, encoders):
    """
    Predict dropout status.
    :param input_data: Pandas DataFrame containing preprocessed data.
    :param model: Trained machine learning model.
    :return: Predictions.
    """
    preprocessed_data = preprocess_data(input_data, encoders)
    predictions = model.predict(preprocessed_data)
    return predictions
