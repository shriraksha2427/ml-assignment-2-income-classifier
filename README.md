# M.Tech (AIML) - Machine Learning Assignment 2

## a. Problem statement
The objective of this assignment is to build, evaluate, compare, and deploy an end-to-end machine learning classification pipeline across five distinct supervised classification algorithms (Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, and Support Vector Machine) to predict annual income levels based on the UCI Adult Census Income dataset, and serve live test-set inference via an interactive Streamlit web application without retraining.

## b. Dataset description
- **Dataset Name**: UCI Adult Census Income Dataset (`adult.data`)
- **Source Repository**: UCI Machine Learning Repository
- **Total Cleaned Instances**: 30,162 instances after handling missing values (`?`)
- **Feature Size**: 14 attributes (mix of continuous numerical features like age, capital gain/loss, hours per week, and encoded categorical features like education, occupation, workclass, and marital status)
- **Target Variable**: Binary classification (`0` = Income $\le \$50\text{K}$, `1` = Income $> \$50\text{K}$)

## c. Github Repository Link
- **GitHub Repository**: [https://github.com/shriraksha2427/ml-assignment-2-income-classifier](https://github.com/shriraksha2427/ml-assignment-2-income-classifier)
- **Live Streamlit App**: [https://<your-app-name>.streamlit.app](https://ml-assignment-2-income-classifier-djlchyhjsvbvepvnaymuzt.streamlit.app/)

## d. Models used & Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8423 | 0.8951 | 0.7254 | 0.5832 | 0.6467 | 0.5512 |
| **Decision Tree** | 0.8512 | 0.8920 | 0.7412 | 0.6120 | 0.6704 | 0.5815 |
| **KNN** | 0.8254 | 0.8710 | 0.6789 | 0.6012 | 0.6378 | 0.5284 |
| **Naive Bayes** | 0.8012 | 0.8540 | 0.6120 | 0.7350 | 0.6678 | 0.5312 |
| **Support Vector Machine** | 0.8540 | 0.9012 | 0.7510 | 0.6085 | 0.6724 | 0.5862 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Standardized scaling via pipeline ensured rapid convergence and robust linear boundary separation, yielding solid overall accuracy (84.23%) and balanced precision-recall. |
| **Decision Tree** | Setting a controlled tree depth (`max_depth=10`) prevented overfitting while capturing non-linear interactions across categorical thresholds, achieving high accuracy (85.12%). |
| **KNN** | Distance-based metrics (`k=7`) performed well after feature normalization, though computation time scaled higher relative to parametric models. |
| **Naive Bayes** | Gaussian Naive Bayes recorded the highest recall (73.50%) by predicting positive classes more broadly, though its precision was lower due to conditional independence assumptions. |
| **Support Vector Machine** | **Overall Winner**. Achieved the highest Accuracy (85.40%), AUC (0.9012), F1 Score (0.6724), and MCC (0.5862) by constructing an optimal maximum-margin hyperplane with the RBF kernel. |
| **Overall, Winner for your dataset?** | **Support Vector Machine (SVM)** |
