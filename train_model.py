import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle

# Load dataset
data = pd.read_csv("data.csv", delimiter=';')

# Encode categorical columns
categorical_columns = [
    'Marital_status', 'Application_mode', 'Course', 'Daytime_evening_attendance',
    'Previous_qualification', 'Nacionality', 'Mothers_qualification', 
    'Fathers_qualification', 'Mothers_occupation', 'Fathers_occupation', 'Gender'
]
label_encoders = {col: LabelEncoder() for col in categorical_columns}

for col in categorical_columns:
    data[col] = label_encoders[col].fit_transform(data[col])

# Split dataset
X = data.drop(columns=['Status'])
y = data['Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and encoders
with open("model.pkl", "wb") as file:
    pickle.dump({"model": model, "encoders": label_encoders}, file)
