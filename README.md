# Medical Expenditure Prediction using MEPS Dataset

This project leverages the **Medical Expenditure Panel Survey (MEPS)** dataset to analyze healthcare expenditure patterns and predict health status outcomes using **Machine Learning techniques**. The goal is to build an end-to-end ML workflow — from data cleaning to model deployment — while addressing **data imbalance** challenges using **ADASYN**.

---

## Project Overview

Healthcare cost prediction and health status assessment are critical for improving medical decision-making, resource allocation, and policy planning.  
In this project, we:

- Performed **data cleaning** and preprocessing on the MEPS dataset  
- Conducted **Exploratory Data Analysis (EDA)** to understand key cost and health trends  
- Handled **data imbalance** using **ADASYN** (Adaptive Synthetic Sampling)  
- Trained multiple **Machine Learning models** to predict health outcomes  
- Evaluated performance using classification metrics  
- Prepared the model for **deployment via Django**

---

## Objectives

- Analyze patterns in medical expenditures  
- Identify key factors influencing healthcare costs  
- Predict an individual's **health status** (`HEALTH` column) using ML models  
- Implement **ADASYN** to improve model fairness and accuracy  

---

## ⚙️ Tech Stack

- **Languages:** Python
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, xgboost  
- **Resampling Technique:** ADASYN  
- **Deployment Framework:** Django *(in progress)*  
- **Version Control:** Git & GitHub  

---

## Workflow

1. **Data Cleaning**
   - Removed duplicates & handled missing values (median imputation)
   - Converted categorical features to numeric (Label/One-Hot Encoding)

2. **EDA**
   - Visualized expenditure trends
   - Identified correlations among key health and cost features

3. **Resampling**
   - Applied **ADASYN** to balance the `HEALTH` target variable

4. **Modeling**
   - Trained models: `XGBClassifier(use_label_encoder=False, eval_metric='logloss')`, Random Forest, Logistic Regression
   - Evaluated using **Accuracy**, **Precision**, **Recall**, and **F1-Score**

5. **Deployment (Ongoing)**
   - Integrating model with a **Django web app** for real-time predictions

---

## Results & Insights

- Initial imbalance led to poor minority class predictions  
- **ADASYN** improved class representation and **F1-scores**  
- Influential features: income, age, number of chronic conditions, insurance status  

---

## Future Improvements

- Integrate with **Django web app** for interactive predictions  
- Apply **Feature Selection** to enhance performance  
- Use **SHAP** or **LIME** for interpretability  
