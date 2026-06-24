#replace "file path" with local dataset path

import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
confusion_matrix,
classification_report
)

from dataset_loader import load_dataset
from feature_engineering import preprocess_data
from ids_model import build_random_forest

# Load dataset

df = load_dataset(file path)

# Preprocess data

X, y, encoder = preprocess_data(df)

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.3,
random_state=42,
stratify=y
)

# Build model

model = build_random_forest()

# Train model

model.fit(X_train, y_train)

# Predictions

y_pred = model.predict(X_test)

# Evaluation

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
y_test,
y_pred,
average="weighted"
)

recall = recall_score(
y_test,
y_pred,
average="weighted"
)

f1 = f1_score(
y_test,
y_pred,
average="weighted"
)

cm = confusion_matrix(y_test, y_pred)

print("\n=== Classification Metrics ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\n=== Confusion Matrix ===")
print(cm)

print("\n=== Detailed Classification Report ===")
print(
classification_report(
y_test,
y_pred,
target_names=encoder.classes_
)
)

# Save trained model

joblib.dump(model, "rf_ids_model.pkl")

print("\nModel saved as rf_ids_model.pkl")
