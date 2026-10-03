# ==========================================
# SCANIA TRUCK APS FAILURE PREDICTION
# Prediction Script
# ==========================================

import os
import joblib
import pandas as pd


# ------------------------------------------
# Project paths
# ------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "random_forest_aps_model.pkl"
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "aps_failure_training_set.csv"
)


# ------------------------------------------
# Check files
# ------------------------------------------

print("Project root:", PROJECT_ROOT)
print("Model path:", MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    print("\nERROR: Model file not found!")
    print("Expected location:")
    print(MODEL_PATH)
    exit()

if not os.path.exists(DATA_PATH):
    print("\nERROR: Dataset file not found!")
    print("Expected location:")
    print(DATA_PATH)
    exit()


# ------------------------------------------
# Load trained model
# ------------------------------------------

model = joblib.load(MODEL_PATH)

print("\nAPS Failure Prediction Model Loaded Successfully!")


# ------------------------------------------
# Load dataset
# ------------------------------------------

df = pd.read_csv(
    DATA_PATH,
    skiprows=20,
    na_values="na"
)

print("Dataset loaded successfully!")


# ------------------------------------------
# Prepare features
# ------------------------------------------

X = df.drop(columns=["class"])

# Remove columns with more than 70% missing values
missing_ratio = X.isnull().mean()

columns_to_drop = missing_ratio[
    missing_ratio > 0.70
].index

X = X.drop(columns=columns_to_drop)

# Fill remaining missing values
X = X.fillna(X.median())


# ------------------------------------------
# Make prediction
# ------------------------------------------

sample = X.iloc[[0]]

prediction = model.predict(sample)[0]

probability = model.predict_proba(sample)[0][1]


# ------------------------------------------
# Display result
# ------------------------------------------

print("\n==========================================")
print("        APS FAILURE PREDICTION")
print("==========================================")

if prediction == 1:
    print("Prediction: APS FAILURE")
else:
    print("Prediction: NORMAL")

print(f"Failure Probability: {probability:.2%}")

print("==========================================")