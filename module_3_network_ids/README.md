This module implements an Intrusion Detection System (IDS) using a Random Forest classifier. It trains on the CICIDS2017 dataset and predicts network attacks.

* dataset_loader.py – Loads the dataset.
* feature_engineering.py – Cleans and preprocesses the data.
* ids_model.py – Creates the Random Forest model.
* train_ids.py – Trains and evaluates the model.
* detect_intrusion.py – Detects intrusions using the trained model.

Dataset

The dataset is not included in this repository due to its size.

Download the CICIDS2017 MachineLearningCSV dataset and update the dataset path in:

* train_ids.py
* detect_intrusion.py


Output

* rf_ids_model.pkl – Trained Random Forest model.
* ai_anomaly_scores.csv – Predicted intrusion results and anomaly scores.
