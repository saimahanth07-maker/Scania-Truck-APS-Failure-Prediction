# Scania Truck APS Failure Prediction

## 📌 Project Overview

The **Scania Truck APS Failure Prediction** project uses machine learning to predict whether a heavy truck is likely to experience an **Air Pressure System (APS) failure**.

The project uses sensor data from the Scania APS dataset and focuses on handling the highly imbalanced classification problem, where normal operating conditions are much more common than failures.

A **class-balanced Random Forest classifier** is trained to identify potential APS failures.

---

## 🎯 Objectives

* Load and analyze the Scania APS dataset.
* Handle missing sensor values.
* Remove features with excessive missing data.
* Handle the highly imbalanced target classes.
* Build a baseline classification model.
* Train a class-balanced Random Forest model.
* Evaluate model performance using precision, recall, F1-score, ROC-AUC, and confusion matrix.
* Save the trained model.
* Create a standalone prediction script.

---

## 📊 Dataset

The project uses the **Scania Truck APS Failure dataset**.

The dataset contains:

* **60,000 training records**
* **170 sensor/features plus the target**
* Target classes:

  * `neg` → Normal operation
  * `pos` → APS failure

The original dataset is highly imbalanced:

| Class   | Records |
| ------- | ------: |
| Normal  |  59,000 |
| Failure |   1,000 |

---

## 🔧 Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset while handling `"na"` values as missing values.
2. Converted the target:

   * `neg` → `0`
   * `pos` → `1`
3. Removed the target column from the feature matrix.
4. Removed features with more than **70% missing values**.
5. Filled remaining missing numerical values using the **median**.
6. Obtained **163 usable features**.
7. Used an **80/20 stratified train-validation split**.

### Dataset Split

| Dataset    | Samples | Features |
| ---------- | ------: | -------: |
| Training   |  48,000 |      163 |
| Validation |  12,000 |      163 |

---

## 🤖 Models

### Baseline Model

A simple baseline model was created that predicts every truck as normal.

The baseline achieved:

* Accuracy: **98.33%**
* Failure Precision: **0.00%**
* Failure Recall: **0.00%**
* Failure F1-score: **0.00%**

Although the accuracy appears high, the baseline failed to identify any APS failures. This demonstrates why accuracy alone is not sufficient for this imbalanced classification problem.

### Random Forest

A Random Forest classifier was trained using:

```python
RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

The `class_weight="balanced"` setting gives greater importance to the minority failure class.

---

## 📈 Model Performance

The Random Forest achieved the following validation results:

| Metric    | Baseline | Random Forest |
| --------- | -------: | ------------: |
| Accuracy  |   98.33% |    **99.18%** |
| Precision |    0.00% |    **74.16%** |
| Recall    |    0.00% |    **77.50%** |
| F1-Score  |    0.00% |    **75.79%** |
| ROC-AUC   |        — |    **99.02%** |

### Confusion Matrix

```text
                  Predicted
                Normal  Failure

Actual Normal     11746      54
Actual Failure       45     155
```

The model correctly identified **155 of the 200 failure cases** in the validation set.

---

## 📊 Visualizations

The project includes:

* Random Forest confusion matrix
* ROC curve
* Top 20 feature importance plot
* Baseline vs Random Forest metric comparison

These visualizations help analyze the model beyond simple accuracy.

---

## 💾 Saved Model

The trained Random Forest model is saved as:

```text
models/random_forest_aps_model.pkl
```

The model can be loaded later using `joblib`.

---

## 🔮 Prediction

A standalone prediction script is provided:

```text
src/predict.py
```

Run it from the project root:

```powershell
python src/predict.py
```

The script loads the saved model, prepares the dataset, and produces a prediction such as:

```text
==========================================
        APS FAILURE PREDICTION
==========================================
Prediction: NORMAL
Failure Probability: 0.00%
==========================================
```

---

## 📁 Project Structure

```text
Scania-Truck-APS-Failure-Prediction/
│
├── data/
│   └── raw/
│       └── aps_failure_training_set.csv
│
├── models/
│   └── random_forest_aps_model.pkl
│
├── notebooks/
│   └── APS Failure Prediction notebook
│
├── src/
│   └── predict.py
│
├── README.md
│
└── requirements.txt
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

Move into the project directory:

```bash
cd Scania-Truck-APS-Failure-Prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## ▶️ Running the Prediction

After installing the dependencies and ensuring the model file exists:

```powershell
python src/predict.py
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* SciPy
* Matplotlib
* Joblib
* Jupyter Notebook
* VS Code
* Git & GitHub

---

## 🚀 Future Improvements

Possible future improvements include:

* Hyperparameter tuning
* Threshold optimization
* Additional imbalance-handling techniques
* Cross-validation
* Model comparison with XGBoost or other classifiers
* Interactive prediction interface
* Deployment as a web application
* Real-time sensor prediction

---

## 👨‍💻 Project Status

**Status: Completed — Machine Learning Prototype**

The project currently includes data preprocessing, model training, evaluation, model persistence, and a standalone prediction script.

