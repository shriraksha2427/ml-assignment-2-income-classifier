# Machine Learning Assignment 2 - Adult Income Classification

## a. Problem statement
The objective of this assignment is to build, evaluate, compare, and deploy an end-to-end machine learning classification pipeline across five distinct supervised classification algorithms (Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, and Support Vector Machine) to predict annual income levels based on the UCI Adult Census Income dataset, and serve live test-set inference via an interactive Streamlit web application without retraining.

## b. Dataset description
- **Dataset Name**: UCI Adult Census Income Dataset (`adult.csv `)
- **Source Repository**: UCI Machine Learning Repository
- **Total Cleaned Instances**: 30,162 instances after handling missing values (`?`)
- **Feature Size**: 14 input features (including age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, and native_country)
- **Target Variable**: Binary classification (`0` : Income $ <= \$50\text{K}$, `1` : Income $> \$50\text{K}$)

## c. Github Repository Link
- **GitHub Repository**: [https://github.com/shriraksha2427/ml-assignment-2-income-classifier](https://github.com/shriraksha2427/ml-assignment-2-income-classifier)
- **Live Streamlit App**: [https://ml-assignment-2-income-classifier-djlchyhjsvbvepvnaymuzt.streamlit.app/](https://ml-assignment-2-income-classifier-djlchyhjsvbvepvnaymuzt.streamlit.app/)

## d. Models used & Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8130 | 0.8231 | 0.7040 | 0.3877 | 0.5000 | 0.4234 |
| **Decision Tree** | 0.8148 | 0.7529 | 0.6123 | 0.6334 | 0.6227 | 0.5001 |
| **KNN** | 0.7783 | 0.6751 | 0.5697 | 0.3304 | 0.4182 | 0.3096 |
| **Naive Bayes** | 0.7998 | 0.8356 | 0.6787 | 0.3227 | 0.4374 | 0.3681 |
| **Support Vector Machine** | 0.7958 | 0.6486 | 0.9839 | 0.1560 | 0.2692 | 0.3461 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Gave good accuracy (81.30%) and a high AUC score (0.8231), but missed many high-income cases, resulting in lower recall (38.77%) and a 0.5000 F1 score. |
| **Decision Tree** | **Overall Winner**. Because tree splits are not affected by large differences in number scales across columns, it achieved the highest Accuracy (81.48%), Recall (63.34%), F1 Score (0.6227), and MCC (0.5001). |
| **KNN** | Had the lowest accuracy (77.83%) and MCC (0.3096) because large numbers in columns like `fnlwgt` and `capital_gain` dominated the distance calculations between data points. |
| **Naive Bayes** | Achieved the highest AUC score (0.8356) and ran very fast, but had low recall (32.27%) because features like education and occupation are closely related rather than independent. |
| **Support Vector Machine** | Had very high precision (98.39%) because it was very cautious when predicting `>50K`, but it missed most of the actual high-income cases, leading to the lowest recall (15.60%), F1 score (0.2692), and AUC (0.6486). |
| **Overall, Winner for your dataset?** | **Decision Tree Classifier** |
