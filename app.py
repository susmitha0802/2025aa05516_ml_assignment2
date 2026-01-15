import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

import seaborn as sns
import matplotlib.pyplot as plt

st.title("ML Assignment 2")
st.write("Model evaluation using uploaded test data")

with st.sidebar:
    st.header("Model Evaluation")

    with st.form("sidebar_form"):
        uploaded_file = st.file_uploader(
            "Upload test CSV file",
            type="csv"
        )

        selected_model = st.selectbox(
            "Select a model",
            (
                "-- Select model --",
                "Logistic Regression",
                "Decision Tree Classifier",
                "K-Nearest Neighbor Classifier",
                "Naive Bayes Classifier",
                "Random Forest",
                "XGBoost",
            ),
        )

        submit = st.form_submit_button("Submit")

if submit:
    if uploaded_file is None:
        st.toast("Please upload a CSV file!", icon = "🔔")
    if selected_model == "-- Select model --":
        st.toast("Please select a model!", icon="🔔")
    if uploaded_file is not None and selected_model != "-- Select model --":
        scaler = joblib.load("model/scaler.pkl")

        models = {
            "Logistic Regression": "model/logistic_regression.pkl",
            "Decision Tree Classifier": "model/decision_tree.pkl",
            "K-Nearest Neighbor Classifier": "model/knn.pkl",
            "Naive Bayes Classifier": "model/naive_bayes.pkl",
            "Random Forest": "model/random_forest.pkl",
            "XGBoost": "model/xgboost.pkl",
        }

        data = pd.read_csv(uploaded_file)

        X = data.drop(["diagnosis"],axis = 1)
        y = data["diagnosis"]

        X_scaled = scaler.transform(X)

        model = joblib.load(models[selected_model])

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_scaled)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
        else:
            y_proba = None
            y_pred = model.predict(X_scaled)

        st.header("Evaluation Metrics")

        metrics_df = pd.DataFrame({
            "Metric": [
                "Accuracy",
                "AUC",
                "Precision",
                "Recall",
                "F1 Score",
                "MCC",
            ],
            "Value": [
                accuracy_score(y, y_pred),
                roc_auc_score(y, y_proba) if y_proba is not None else np.nan,
                precision_score(y, y_pred),
                recall_score(y, y_pred),
                f1_score(y, y_pred),
                matthews_corrcoef(y, y_pred),
            ]
        })
        
        st.dataframe(metrics_df, hide_index=True, key="metrics")

        st.header("Confusion Matrix")

        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)