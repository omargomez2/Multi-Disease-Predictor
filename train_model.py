import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Generate synthetic dataset
np.random.seed(42)
n_samples = 1000

features = {
    "Hemoglobin": np.random.normal(14, 1.5, n_samples),
    "WBC": np.random.normal(7000, 1500, n_samples),
    "Platelets": np.random.normal(250000, 50000, n_samples),
    "Glucose": np.random.normal(100, 15, n_samples),
    "Urea": np.random.normal(30, 5, n_samples),
    "Creatinine": np.random.normal(1.0, 0.2, n_samples),
    "Sodium": np.random.normal(140, 4, n_samples),
    "Potassium": np.random.normal(4.2, 0.5, n_samples),
    "Chloride": np.random.normal(100, 3, n_samples),
    "Calcium": np.random.normal(9.5, 0.5, n_samples),
    "Bilirubin": np.random.normal(1.0, 0.3, n_samples),
    "ALT": np.random.normal(30, 10, n_samples),
    "AST": np.random.normal(30, 10, n_samples),
    "Alkaline_Phosphatase": np.random.normal(100, 30, n_samples),
    "CRP": np.random.normal(5, 2, n_samples)
}
X = pd.DataFrame(features)

diseases = [
    "Anemia", "Diabetes", "Kidney Disease", "Liver Disease", "Hypertension",
    "Leukemia", "Infection", "Cancer", "Thyroid Disorder", "Autoimmune Disorder",
    "Hepatitis", "Pancreatitis", "Gallbladder Disease", "Gastritis", "Ulcer"
]
y = np.random.choice(diseases, n_samples)

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model and label encoder
joblib.dump(clf, "disease_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Model and encoder saved successfully.")