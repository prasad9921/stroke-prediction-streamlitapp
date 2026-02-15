# Stroke Prediction & Analysis System - ML Assignment 2

## 1. Project Overview
This project implements an end-to-end Machine Learning pipeline to predict the likelihood of a patient suffering a stroke. It includes Exploratory Data Analysis (EDA), the implementation of 6 different classification models, and an interactive web application built with Streamlit for real-time inference and insight visualization.

**Problem Statement:** To develop a predictive model that can classify whether a patient is at risk of stroke based on demographic data (gender, age, residence), health conditions (hypertension, heart disease), and lifestyle factors (smoking status, BMI).

## 2. Dataset Description
- **Dataset Name:** Stroke Prediction Dataset
- **Source:** [Kaggle / https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset]
- **Type:** Binary Classification (Stroke: 1, No Stroke: 0)
- **Instances:** 5110 rows
- **Features:** 12 columns (11 predictors, 1 target)
- **Categorical:** gender, ever_married, work_type, Residence_type, smoking_status
- **Numerical:** age, avg_glucose_level, bmi
- **Binary Health Indicators:** hypertension, heart_disease

**Key Preprocessing Steps:**
- **Imputation:** Missing bmi values were filled using the median.
- **Transformation:** avg_glucose_level was log-transformed to reduce skewness.
- **Encoding:** One-Hot Encoding for categorical variables; Label Encoding for binary features.
- **Balancing:** Addressed extreme class imbalance (only ~5% positive cases) using Class Weighting and SMOTE techniques.

## 3. Project Structure
```
Root/
├── data/
│   └── healthcare-dataset-stroke-data.csv   # Original dataset
├── model/
│   ├── logistic_regression.py               # Training script for Logistic Regression
│   ├── decision_tree.py                     # Training script for Decision Tree
│   ├── knn_classifier.py                    # Training script for k-NN
│   ├── naive_bayes.py                       # Training script for Naive Bayes
│   ├── random_forest.py                     # Training script for Random Forest
│   └── xgboost.py                           # Training script for XGBoost
├── app/
│   ├── app.py                               # Main Streamlit application entry point
│   ├── data_insights.py                     # Module for EDA visualizations
│   ├── model_insights.py                    # Module for model metrics & curves
│   └── inference.py                         # Module for real-time prediction
│   ├── model_assets/
│   │   ├── preprocessor.joblib              # Saved transformation pipeline
│   │   ├── model_metadata.json              # Metrics & training history for the app
│   │   └── *.joblib                         # Saved model weights (6 files)
├── Fullassignment.ipynb                     # Comprehensive notebook (EDA + Training)
├── requirements.txt                         # Dependency list
└── README.md                                # Project documentation
```

## 4. Models Implemented & Evaluation
We implemented six classification models. Due to the high class imbalance, we prioritized **Recall** (minimizing false negatives) and **F1-Score** over pure Accuracy.

### Performance Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.7211 | 0.7939 | 0.1293 | 0.82 | 0.2234 | 0.25 |
| Decision Tree | 0.8405 | 0.6995 | 0.1576 | 0.52 | 0.2419 | 0.221 |
| kNN (with SMOTE) | 0.8102 | 0.6337 | 0.0909 | 0.32 | 0.1416 | 0.0888 |
| Naive Bayes | 0.3425 | 0.7846 | 0.0681 | 0.98 | 0.1273 | 0.1369 |
| Random Forest | 0.9491 | 0.7812 | 0 | 0 | 0 | -0.01 |
| XGBoost (Weighted) | 0.908 | 0.8094 | 0.2027 | 0.3 | 0.2419 | 0.1992 |

### Observations

| ML Model Name | Observation about model performance |
|---------------|------------------------------------|
| Logistic Regression | Showed excellent Recall (~82%), meaning it catches most stroke cases, but at the cost of many False Positives (low Precision). Good baseline. |
| Decision Tree | Prone to overfitting. While accuracy is high, it struggled to generalize on the minority class compared to ensemble methods. |
| kNN | Required SMOTE to perform reasonably. It struggled with the high dimensionality of the one-hot encoded features. |
| Naive Bayes | Achieved the highest Recall (~98%) but the lowest Precision. It is very "trigger happy" in predicting strokes, which can be useful for initial screening. |
| Random Forest | High accuracy but very poor Recall (missed many strokes) despite class weighting. It favored the majority class too heavily. |
| XGBoost | **Best Overall Performer.** It achieved the best balance (highest F1 and MCC scores) and a strong AUC, successfully managing the trade-off between precision and recall. |

## 5. How to Run the Project

### Prerequisites
Ensure you have Python 3.9+ installed.

### Setup Instructions

**Clone the Repository:**
```bash
git clone <your-repo-link>
cd <repo-name>
```

**Create a Virtual Environment:**
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

**Install Dependencies:**
```bash
pip install -r requirements.txt
```

**Run the Streamlit App:**
```bash
streamlit run app/app.py
```

## 6. App Walkthrough
The application is divided into three functional tabs:

**📊 Data Insights:**
- **View Data:** Toggle between the default dataset or upload your own CSV.
- **Univariate Analysis:** Select any feature (e.g., age, bmi) to view its distribution (Histogram for numerical, Countplot for categorical).
- **Bivariate Analysis:** Choose X and Y axes to visualize relationships (e.g., "Age vs. Stroke" or "Hypertension vs. Stroke") using interactive plots.

**🤖 Model Insights:**
- Select a model from the top navigation (pills).
- View the specific Hyperparameters used during training.
- Analyze the Training Curve (Accuracy/Loss over epochs) to check for overfitting.
- See the detailed Confusion Matrix and performance metrics.

**🩺 Inference Lab:**
- Input patient details using the sidebar form (Age, Glucose Level, BMI, Smoking Status, etc.).
- Select the model you wish to use for prediction (e.g., XGBoost).
- Click "Analyze Patient Data" to get a real-time risk assessment (Low Risk / High Risk) with the probability score.

---

**Submitted by:** Yerramsetty Nishitha

**ID:** 2024DC04261