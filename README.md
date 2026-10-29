# Machine Learning Assignment 2 - Adult Income Classification

## a. Problem statement
The objective of this assignment is to build, evaluate, compare, and deploy an end-to-end machine learning classification pipeline across five distinct supervised classification algorithms (Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, and Support Vector Machine) to predict annual income levels based on the UCI Adult Census Income dataset, and serve live test-set inference via an interactive Streamlit web application without retraining.

## b. Dataset description
- **Dataset Name**: UCI Adult Census Income Dataset (`adult.csv`)
- **Source Repository**: UCI Machine Learning Repository
- **Total Cleaned Instances**: 30,162 instances after handling missing values (`?`)
- **Feature Size**: 14 input features (including age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, and native_country)
- **Target Variable**: Binary classification (`0` : Income <= $50\text{K}$, `1` : Income > $50\text{K}$)

## c. Github Repository Link
- **GitHub Repository**: [https://github.com/shriraksha2427/ml-assignment-2-income-classifier](https://github.com/shriraksha2427/ml-assignment-2-income-classifier)
- **Live Streamlit app**: [https://ml-assignment-2-income-classifier-djlchyhjsvbvepvnaymuzt.streamlit.app/](https://ml-assignment-2-income-classifier-djlchyhjsvbvepvnaymuzt.streamlit.app/)

## d. Models used & Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8175 | 0.8501 | 0.7135 | 0.4461 | 0.5490 | 0.4613 |
| **Decision Tree** | 0.8505 | 0.8892 | 0.7504 | 0.5985 | 0.6659 | 0.5772 |
| **KNN** | 0.8221 | 0.8614 | 0.6631 | 0.5806 | 0.6191 | 0.5056 |
| **Naive Bayes** | 0.7865 | 0.8289 | 0.6569 | 0.2983 | 0.4103 | 0.3368 |
| **Support Vector Machine** | 0.8414 | 0.8888 | 0.7466 | 0.5493 | 0.6329 | 0.5450 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved solid overall accuracy (81.75%) and AUC (0.8501) with scaled numerical features, though recall remained moderate at 44.61%. |
| **Decision Tree** | **Overall Winner**. With max depth tuned properly, it achieved the highest Accuracy (85.05%), AUC (0.8892), Precision (75.04%), Recall (59.85%), F1 Score (66.59%), and MCC (57.72%). |
| **KNN** | Performed competitively after standard scaling, yielding 82.21% accuracy and a 0.6191 F1 score. |
| **Naive Bayes** | Recorded a solid AUC of 0.8289 and fast computation, but exhibited lower recall (29.83%) due to feature conditional independence assumptions. |
| **Support Vector Machine** | Demonstrated strong performance with 84.14% accuracy and 0.8888 AUC, closely following the Decision Tree by leveraging maximum-margin hyperplanes. |
| **Overall, Winner for your dataset?** | **Decision Tree Classifier** |
