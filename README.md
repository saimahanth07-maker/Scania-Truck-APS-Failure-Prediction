# Scania Truck APS Failure Prediction

A machine learning project for predicting **Air Pressure System (APS) failures in Scania heavy trucks** using sensor data and a Random Forest classification model.

## 📌 Project Overview

The Air Pressure System (APS) is an important component in heavy trucks. A failure in the APS can affect vehicle operation and may require maintenance.

This project uses machine learning to analyze truck sensor measurements and predict whether a truck is likely to experience an APS failure.

The system includes:

* Data preprocessing
* Missing-value handling
* Class-imbalance analysis
* Machine learning model training
* Random Forest classification
* Model evaluation
* Prediction probability estimation
* Flask web application
* Model performance dashboard
* Confusion matrix visualization
* ROC curve visualization
* Feature importance visualization

---

## 🎯 Objectives

The main objectives of this project are:

1. Analyze the Scania APS sensor dataset.
2. Clean and preprocess the sensor data.
3. Handle the highly imbalanced target classes.
4. Train a machine learning model for APS failure prediction.
5. Evaluate the model using multiple performance metrics.
6. Build a web-based prediction interface.
7. Display model performance and visualizations through a dashboard.

---

## 📊 Dataset

The project uses the **Scania APS Failure dataset**.

### Dataset characteristics

| Property            |   Value |
| ------------------- | ------: |
| Total records       |  60,000 |
| Original features   |     170 |
| Target column       | `class` |
| Normal records      |  59,000 |
| APS failure records |   1,000 |
| Missing values      | 850,015 |

The target variable contains two classes:

* `neg` → Normal truck
* `pos` → APS failure

The dataset is highly imbalanced, with significantly fewer failure cases than normal cases.

---

## 🔧 Data Preprocessing

The following preprocessing steps were performed:

### 1. Dataset loading

The raw Scania dataset was loaded using Pandas.

### 2. Metadata removal

The dataset contains metadata/header information, so the appropriate rows were skipped during loading.

### 3. Target conversion

The target class was converted into numerical values:

```text
neg → 0
pos → 1
```

### 4. Missing-value conversion

The string value `na` was converted into proper missing values.

### 5. Numeric conversion

Sensor features were converted into numerical data types.

### 6. High-missingness feature removal

Features with more than **70% missing values** were removed.

### 7. Median imputation

Remaining missing values were replaced using the median value of each feature.

### 8. Stratified train-validation split

The processed dataset was divided into training and validation sets while preserving the class distribution.

```text
Training set:   48,000 records
Validation set: 12,000 records
```

---

## 🤖 Machine Learning Model

### Random Forest Classifier

The main prediction model is a **Random Forest Classifier**.

Configuration:

```python
RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

The `class_weight="balanced"` parameter was used to help the model handle the strong class imbalance in the dataset.

---

## 📈 Model Performance

The Random Forest model was evaluated on the validation dataset.

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 99.18% |
| Precision | 74.16% |
| Recall    | 77.50% |
| F1 Score  | 75.79% |
| ROC-AUC   | 99.02% |

### Confusion Matrix

The validation confusion matrix was:

```text
[[11746    54]
 [   45   155]]
```

This represents:

* True Negatives: **11,746**
* False Positives: **54**
* False Negatives: **45**
* True Positives: **155**

The model therefore identifies a substantial portion of the APS failure cases while maintaining high overall accuracy.

---

## 📊 Model Visualizations

The project includes the following visualizations:

### Confusion Matrix

Shows the number of correct and incorrect predictions for each class.

### ROC Curve

Shows the relationship between the true-positive rate and false-positive rate at different classification thresholds.

### Feature Importance

Displays the top 20 sensor features contributing to the Random Forest model.

These visualizations are available in:

```text
app/static/
├── confusion_matrix.png
├── roc_curve.png
└── feature_importance.png
```

---

## 🌐 Web Application

A Flask-based web application was developed to make the trained model accessible through a browser.

Run the application using:

```powershell
python app\app.py
```

Then open:

```text
http://127.0.0.1:5000
```

### Web application features

The application provides:

* Truck record selection
* APS failure prediction
* Failure probability
* Actual dataset result
* Model performance metrics
* Confusion matrix
* ROC curve
* Feature importance chart
* Responsive user interface

---

## 🔮 Prediction Output

For a selected truck record, the system displays:

```text
Prediction: NORMAL
Failure Probability: 0.00%
Actual Result: NORMAL
```

or:

```text
Prediction: APS FAILURE
Failure Probability: 100.00%
Actual Result: APS FAILURE
```

The probability is generated using the Random Forest model's `predict_proba()` function.

---

## 📁 Project Structure

```text
Scania-Truck-APS-Failure-Prediction/
│
├── app/
│   ├── app.py
│   │
│   ├── static/
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   └── feature_importance.png
│   │
│   └── templates/
│       └── index.html
│
├── data/
│   └── raw/
│       └── aps_failure_training_set.csv
│
├── models/
│   └── random_forest_aps_model.pkl
│
├── notebooks/
│   └── 01_Data_Collection.ipynb
│
├── src/
│   └── predict.py
│
├── README.md
│
└── requirements.txt
```

---

## 💻 Technologies Used

### Programming Language

* Python 3.13

### Libraries

* Pandas
* NumPy
* Scikit-learn
* SciPy
* Joblib
* Matplotlib
* Jupyter
* Flask

### Development Tools

* Visual Studio Code
* Git
* GitHub

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/saimahanth07-maker/Scania-Truck-APS-Failure-Prediction.git
```

Navigate into the project:

```bash
cd Scania-Truck-APS-Failure-Prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## ▶️ Running the Prediction Script

The standalone prediction script can be executed using:

```powershell
python src\predict.py
```

The script loads the trained Random Forest model and performs a prediction using a sample truck record.

---

## 🌐 Running the Flask Application

Start the web application:

```powershell
python app\app.py
```

Open the application in a browser:

```text
http://127.0.0.1:5000
```

---

## 🧠 Model File

The trained model is stored at:

```text
models/random_forest_aps_model.pkl
```

The Flask application loads this model using Joblib.

---

## 🚀 Future Improvements

Possible future improvements include:

* Hyperparameter tuning
* XGBoost or LightGBM comparison
* Advanced class-imbalance techniques such as SMOTE
* Cross-validation
* Threshold optimization
* SHAP-based model explainability
* Real-time sensor-data integration
* Cloud deployment
* Database integration
* Docker deployment
* Automated model retraining
* Monitoring model performance over time

---

## 📌 Project Status

**Status: Completed**

The project currently includes:

* ✅ Dataset preprocessing
* ✅ Missing-value handling
* ✅ Class-imbalance handling
* ✅ Random Forest model
* ✅ Model evaluation
* ✅ Saved trained model
* ✅ Standalone prediction script
* ✅ Flask web application
* ✅ Prediction interface
* ✅ Model performance dashboard
* ✅ Confusion matrix
* ✅ ROC curve
* ✅ Feature importance visualization
* ✅ GitHub repository

---

## 👨‍💻 Author

**Sai Mahanth**

Scania Truck APS Failure Prediction — Machine Learning Project
