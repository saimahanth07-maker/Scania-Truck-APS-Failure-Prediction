from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Project root directory
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Model path
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "random_forest_aps_model.pkl"
)

# Dataset path
DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "aps_failure_training_set.csv"
)

# Load trained model
model = joblib.load(MODEL_PATH)

# Load dataset
df = pd.read_csv(DATA_PATH, skiprows=20)

# Keep original class for displaying actual result
actual_class = df["class"].copy()

# Convert class to numeric
df["class"] = df["class"].map({
    "neg": 0,
    "pos": 1
})

# Remove target column
X = df.drop(columns=["class"])

# Convert "na" strings to proper missing values
X = X.replace("na", pd.NA)

# Convert all feature columns to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Remove columns with more than 70% missing values
missing_ratio = X.isnull().mean()

X = X.loc[:, missing_ratio <= 0.70]

# Fill remaining missing values using median
X = X.fillna(X.median())
# Match trained model feature order
X = X[model.feature_names_in_]


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    probability = None
    actual = None
    record_number = None
    error = None

    if request.method == "POST":

        try:
            record_number = int(request.form["record_number"])

            # Check valid record number
            if record_number < 1 or record_number > len(X):
                raise ValueError(
                    f"Please enter a record number between 1 and {len(X)}."
                )

            # Convert to zero-based index
            index = record_number - 1

            # Select truck record
            sample_data = X.iloc[[index]]

            # Make prediction
            predicted_class = model.predict(sample_data)[0]

            # Failure probability
            probability = model.predict_proba(sample_data)[0][1] * 100

            # Prediction text
            if predicted_class == 1:
                prediction = "APS FAILURE"
            else:
                prediction = "NORMAL"

            # Actual class from dataset
            if actual_class.iloc[index] == "pos":
                actual = "APS FAILURE"
            else:
                actual = "NORMAL"

        except ValueError as e:
            error = str(e)

        except Exception as e:
            error = f"An unexpected error occurred: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=round(probability, 2) if probability is not None else None,
        actual=actual,
        record_number=record_number,
        error=error,
        total_records=len(X)
    )


if __name__ == "__main__":
    app.run(debug=True)