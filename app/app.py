from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ============================================================
# PROJECT PATHS
# ============================================================

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


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load(MODEL_PATH)


# ============================================================
# LOAD DATASET
# ============================================================

# Scania dataset contains metadata/header rows,
# so skip the first 20 rows.
df = pd.read_csv(
    DATA_PATH,
    skiprows=20
)


# ============================================================
# KEEP ORIGINAL CLASS
# ============================================================

# Used later to display the actual result
actual_class = df["class"].copy()


# ============================================================
# CONVERT TARGET COLUMN
# ============================================================

df["class"] = df["class"].map({
    "neg": 0,
    "pos": 1
})


# ============================================================
# REMOVE TARGET COLUMN
# ============================================================

X = df.drop(
    columns=["class"]
)


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

# Convert "na" strings to proper missing values
X = X.replace(
    "na",
    pd.NA
)


# Convert all feature columns to numeric
X = X.apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# REMOVE HIGH-MISSINGNESS FEATURES
# ============================================================

missing_ratio = X.isnull().mean()

X = X.loc[
    :,
    missing_ratio <= 0.70
]


# ============================================================
# FILL REMAINING MISSING VALUES
# ============================================================

X = X.fillna(
    X.median()
)


# ============================================================
# MATCH MODEL FEATURE ORDER
# ============================================================

X = X[
    model.feature_names_in_
]


# ============================================================
# HOME ROUTE
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def home():

    # Default values
    prediction = None
    probability = None
    actual = None
    record_number = None
    error = None

    # ========================================================
    # MODEL PERFORMANCE METRICS
    # ========================================================

    metrics = {
        "accuracy": 99.18,
        "precision": 74.16,
        "recall": 77.50,
        "f1_score": 75.79,
        "roc_auc": 99.02
    }

    # ========================================================
    # HANDLE PREDICTION REQUEST
    # ========================================================

    if request.method == "POST":

        try:

            # Get record number from form
            record_number = int(
                request.form["record_number"]
            )

            # ------------------------------------------------
            # CHECK RECORD NUMBER
            # ------------------------------------------------

            if (
                record_number < 1
                or record_number > len(X)
            ):
                raise ValueError(
                    f"Please enter a record number between 1 and {len(X)}."
                )

            # ------------------------------------------------
            # CONVERT TO ZERO-BASED INDEX
            # ------------------------------------------------

            index = record_number - 1

            # ------------------------------------------------
            # SELECT TRUCK RECORD
            # ------------------------------------------------

            sample_data = X.iloc[
                [index]
            ]

            # ------------------------------------------------
            # MAKE PREDICTION
            # ------------------------------------------------

            predicted_class = model.predict(
                sample_data
            )[0]

            # ------------------------------------------------
            # FAILURE PROBABILITY
            # ------------------------------------------------

            probability = (
                model.predict_proba(sample_data)[0][1]
                * 100
            )

            # ------------------------------------------------
            # CONVERT PREDICTION TO TEXT
            # ------------------------------------------------

            if predicted_class == 1:

                prediction = "APS FAILURE"

            else:

                prediction = "NORMAL"

            # ------------------------------------------------
            # GET ACTUAL DATASET RESULT
            # ------------------------------------------------

            if actual_class.iloc[index] == "pos":

                actual = "APS FAILURE"

            else:

                actual = "NORMAL"

        except ValueError as e:

            error = str(e)

        except Exception as e:

            error = (
                f"An unexpected error occurred: {str(e)}"
            )

    # ========================================================
    # SEND DATA TO HTML TEMPLATE
    # ========================================================

    return render_template(
        "index.html",

        prediction=prediction,

        probability=(
            round(probability, 2)
            if probability is not None
            else None
        ),

        actual=actual,

        record_number=record_number,

        error=error,

        total_records=len(X),

        metrics=metrics
    )


# ============================================================
# RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )