#replace "file path" with dataset path

import joblib
import pandas as pd

from dataset_loader import load_dataset
from feature_engineering import preprocess_data

# Load trained model

model = joblib.load("rf_ids_model.pkl")

# Load data to analyze

df = load_dataset(file path)

# Preprocess data

X, y, encoder = preprocess_data(df)

# Predict attack classes

predictions = model.predict(X)

# Predict probabilities

probabilities = model.predict_proba(X)

# Highest probability for each prediction

anomaly_scores = probabilities.max(axis=1)

# Create output dataframe

results = pd.DataFrame({
"Predicted_Class": encoder.inverse_transform(predictions),
"AnomalyScore": anomaly_scores
})

# Save for Fusion Analytics module

results.to_csv("ai_anomaly_scores.csv", index=False)

print("Detection complete.")
print("Results saved to ai_anomaly_scores.csv")

print("\nSample Predictions:")
print(results.head())
