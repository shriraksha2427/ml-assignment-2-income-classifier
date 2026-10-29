import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Page Configuration
st.set_page_config(
    page_title="Adult Income Classification Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 M.Tech AIML — ML Assignment 2: Classification Dashboard")
st.markdown("**End-to-End ML Deployment Workflow**: Model Evaluation & Streamlit Inference")

# Model File Mapping
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/svm.pkl"
}

@st.cache_resource
def load_saved_models():
    loaded = {}
    for name, path in MODEL_FILES.items():
        if os.path.exists(path):
            with open(path, "rb") as f:
                loaded[name] = pickle.load(f)
    return loaded

models_dict = load_saved_models()

def evaluate_model(model, X_eval, y_eval):
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else [0]*len(y_pred)
    
    metrics = {
        "Accuracy": accuracy_score(y_eval, y_pred),
        "AUC": roc_auc_score(y_eval, y_prob) if len(np.unique(y_eval)) > 1 else 0.0,
        "Precision": precision_score(y_eval, y_pred, zero_division=0),
        "Recall": recall_score(y_eval, y_pred, zero_division=0),
        "F1": f1_score(y_eval, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_eval, y_pred)
    }
    return metrics, y_pred

# Sidebar Configuration (Mandatory Features a & b)
st.sidebar.header("⚙️ App Controls")
uploaded_file = st.sidebar.file_uploader("1. Upload Test Dataset (CSV)", type=["csv"])
model_choice = st.sidebar.selectbox("2. Select Classification Model", list(MODEL_FILES.keys()))

st.sidebar.markdown("---")
st.sidebar.info(f"Loaded Models: {len(models_dict)} / 5 available.")

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    
    if "income" not in test_df.columns:
        st.error("Uploaded CSV must contain the target column (`income`)!")
    else:
        X_test = test_df.drop(columns=["income"])
        y_test = test_df["income"]
        
        active_model = models_dict[model_choice]
        metrics, preds = evaluate_model(active_model, X_test, y_test)
        
        # Tabs for Layout
        tab1, tab2, tab3 = st.tabs([
            f"📈 {model_choice} Metrics",
            "🏆 Cross-Model Comparison",
            "🗂️ Test Data Preview"
        ])
        
        with tab1:
            st.subheader(f"Performance Metrics: {model_choice}")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
            c2.metric("AUC Score", f"{metrics['AUC']:.4f}")
            c3.metric("Precision", f"{metrics['Precision']:.4f}")
            c4.metric("Recall", f"{metrics['Recall']:.4f}")
            c5.metric("F1 Score", f"{metrics['F1']:.4f}")
            c6.metric("MCC Score", f"{metrics['MCC']:.4f}")
            
            st.markdown("---")
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, preds)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("True Label")
                st.pyplot(fig)
                
            with col_b:
                st.subheader("Classification Report")
                report = classification_report(y_test, preds, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
                
        with tab2:
            st.subheader("Benchmark Comparison Across All Models")
            comparison_rows = []
            for m_name, m_obj in models_dict.items():
                m_metrics, _ = evaluate_model(m_obj, X_test, y_test)
                row = {"ML Model Name": m_name}
                row.update({k: round(v, 4) for k, v in m_metrics.items()})
                comparison_rows.append(row)
                
            comp_df = pd.DataFrame(comparison_rows)
            st.dataframe(
                comp_df.style.highlight_max(
                    subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
                    color="#d4edda"
                ),
                use_container_width=True
            )
            
        with tab3:
            st.subheader("Uploaded Test Dataset Inspector")
            st.dataframe(test_df.head(15), use_container_width=True)
else:
    st.info("👈 Please upload your `test_data.csv` file via the sidebar to evaluate models without retraining.")