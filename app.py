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

import streamlit as st

from style import style

st.set_page_config(
    page_title="ML Models Comparison",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

style()

with st.sidebar:
    st.header("Model Testing")

    with st.form("sidebar_form"):
        uploaded_file = st.file_uploader(
            label="Upload test CSV file",
            type="csv",
            key="upload_file"
        )

        selected_models = st.multiselect(
            label="Select models",
            options=[
                "Logistic Regression",
                "Decision Tree Classifier",
                "K-Nearest Neighbor Classifier",
                "Naive Bayes Classifier",
                "Random Forest",
                "XGBoost",
            ],
            key="select_model",
            placeholder="Select models"
        )

        submit = st.form_submit_button(
            label="Submit", 
            key="submit", 
            type="primary", 
            width="stretch"
        )

    file_path = "./data/test_data.csv"

    with st.container(key="download_container"):
        with open(file_path, "rb") as file:
            st.download_button(
                label="Download Test Data", 
                data=file,
                file_name="test_data.csv", 
                mime="text/csv",
                key="download2",
                icon=":material/download:"
            )

st.title("ML Models Evaluation")

st.write("")

if "submitted_successfully" not in st.session_state:
    st.session_state.submitted_successfully = False

if submit:
    if uploaded_file is None:
        st.session_state["submitted_successfully"] = False
        st.toast("Please upload a CSV file!", icon = "🔔")

    if len(selected_models) == 0:
        st.session_state["submitted_successfully"] = False
        st.toast("Please select a model!", icon="🔔")

    if uploaded_file is not None and len(selected_models) != 0:
        st.session_state["submitted_successfully"] = True
        
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

        for selected_model in selected_models:
            st.header("Model: " + selected_model)

            model = joblib.load(models[selected_model])

            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_scaled)[:, 1]
                y_pred = (y_proba >= 0.5).astype(int)
            else:
                y_proba = None
                y_pred = model.predict(X_scaled)

            with st.container():
            
                col1, col2 = st.columns(2, gap="large")

                with col1:
                    st.subheader("Evaluation Metrics")

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

                    st.dataframe(
                        data=metrics_df,
                        hide_index=True
                    )

                with col2:

                    st.subheader("Confusion Matrix")

                    cm = confusion_matrix(y, y_pred)

                    fig, ax = plt.subplots(figsize=(3, 2))
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

if not st.session_state.submitted_successfully:
    
    st.write(
            """
            This application is designed to perform **Breast Cancer Classification**
            using multiple machine learning models.

            Users can upload a test dataset, select one or more classification models, and evaluate model performance (A sample test dataset is provided for reference and is available for download!).

            The application displays standard evaluation metrics along with the
            corresponding confusion matrix for each selected model.

            Please select at least one model from the sidebar and click **Submit**
            to view the results.
            """
        )
    
    st.subheader("Dataset Feature Descriptions")

    data = {
        "Feature": [
            "diagnosis",
            "id",
            "radius_mean",
            "texture_mean",
            "perimeter_mean",
            "area_mean",
            "smoothness_mean",
            "compactness_mean",
            "concavity_mean",
            "concave points_mean",
            "symmetry_mean",
            "fractal_dimension_mean",
            "radius_se",
            "texture_se",
            "perimeter_se",
            "area_se",
            "smoothness_se",
            "compactness_se",
            "concavity_se",
            "concave points_se",
            "symmetry_se",
            "fractal_dimension_se",
            "radius_worst",
            "texture_worst",
            "perimeter_worst",
            "area_worst",
            "smoothness_worst",
            "compactness_worst",
            "concavity_worst",
            "concave points_worst",
            "symmetry_worst",
            "fractal_dimension_worst"
        ],
        "Description": [
            "The diagnosis of breast tissues (M = malignant, B = benign)",
            "ID number (removed during preprocessing as it is non-informative)",
            "mean of distances from center to points on the perimeter",
            "standard deviation of gray-scale values",
            "mean size of the core tumor",
            "mean area of the core tumor",
            "mean of local variation in radius lengths",
            "mean of (perimeter² / area − 1)",
            "mean severity of concave portions of the contour",
            "mean for number of concave portions of the contour",
            "mean symmetry of the tumor",
            "mean coastline approximation − 1",
            "standard error for radius",
            "standard error for texture",
            "standard error for perimeter",
            "standard error for area",
            "standard error for smoothness",
            "standard error for perimeter^2 / area - 1.0",
            "standard error for severity of concave portions of the contour",
            "standard error for number of concave portions of the contour",
            "standard error for symmetry of the tumor",
            "standard error for coastline approximation − 1",
            "largest mean value for mean of distances from center to points on the perimeter",
            "largest mean value for standard deviation of gray-scale values",    
            "largest mean value for mean size of the core tumor",
            "largest mean value for mean area of the core tumor",
            "largest mean value for mean of local variation in radius lengths",
            "largest mean value for mean of (perimeter² / area − 1)",
            "largest mean value for mean severity of concave portions of the contour",
            "largest mean value for mean for number of concave portions of the contour",
            "largest mean value for mean symmetry of the tumor",
            "largest mean value for mean coastline approximation − 1"
        ]
    }

    df = pd.DataFrame(data)
    st.dataframe(df, hide_index=True)

