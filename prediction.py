import pickle
from preprocess import preprocess_data

def load_model():
    """Load the trained model and encoders from files."""
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    with open("encoders.pkl", "rb") as file:
        encoders = pickle.load(file)
    return model, encoders

def predict(data):
    """Make predictions on input data."""
    model, encoders = load_model()
    processed_data = preprocess_data(data, encoders)
    predictions = model.predict(processed_data)
    return predictions

