import streamlit as st

from src.ml.sklearn import loadModel, getVersions
from src.ml.models import ModelType
from src.ml.evaluation import loadEvaluations
from src.graphing.modelPredictions import graphFeatureImportance, graphRandomForestClassifierConfidence
from src.graphing.featureImportances import graphLinearRegression, graphGradientBoostingRegressor, graphLinearModels
from sklearn.metrics import classification_report
import logging
import matplotlib.pyplot as plt
import numpy as np

# Configure logging 
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    handlers=[
                        logging.FileHandler('app.log', encoding='utf-8'),
                        logging.StreamHandler()
                    ])

# Create a logger
logger = logging.getLogger("main")

Y_test = np.load('testData.npy')

@st.cache_resource
def load_model(selected_model_type, selected_version):
    model, version = loadModel(selected_model_type, selected_version)
    return model

# Make predictions
Y_test_preds = np.load('predictions.npy')

## Dashboard
st.set_page_config(layout="wide")
st.title(":green[Bublify] :green[:squid:] :minidisc: :eggplant:")
st.markdown("Song Recommendations based on different LLM Models")

# Create a sidebar for Filters
st.sidebar.title("Dropdown Filters")
selected_model_type = st.sidebar.selectbox("Select Model Type", [member.name for member in ModelType])
selected_version = st.sidebar.selectbox("Select Version", getVersions())

# Load Evals
evals = loadEvaluations(selected_version)
model_names = [eval["name"] for eval in evals]
mse_values = [eval["Mean Squared Error (MSE)"] for eval in evals]
mae_values = [eval["Mean Absolute Error (MAE)"] for eval in evals]

# Load the model
model = load_model(selected_model_type, selected_version)

tab1, tab2, tab3, tab4 = st.tabs(["Model Comparisons", "Feature Importances", "Classification Summaries","Fine Tuning/Predicting"])

with tab1:
    st.header("Overview of Trained Models")
    tab1_col1, tab1_col2 = st.columns(2)
    with tab1_col1:
        # Sort the models based on MSE values in ascending order
        sorted_models = sorted(zip(model_names, mse_values), key=lambda x: x[1])
        sorted_model_names, sorted_mse_values = zip(*sorted_models)

        fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
        ax.barh(sorted_model_names, sorted_mse_values)
        ax.set_ylabel("Model")
        ax.set_xlabel("Mean Squared Error (MSE)")
        ax.set_title("Comparison of Mean Squared Error (MSE) across Models")
        ax.invert_yaxis()  # Invert the y-axis to show the best model at the top

        # Display the graph in Streamlit
        st.pyplot(fig)
    with tab1_col2:
        # Sort the models based on MAE values in ascending order
        sorted_models = sorted(zip(model_names, mae_values), key=lambda x: x[1])
        sorted_model_names, sorted_mae_values = zip(*sorted_models)

        fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
        ax.barh(sorted_model_names, sorted_mae_values)
        ax.set_ylabel("Model")
        ax.set_xlabel("Mean Absolute Error (MAE)")
        ax.set_title("Comparison of Mean Absolute Error (MAE) across Models")
        ax.invert_yaxis()  # Invert the y-axis to show the best model at the top

        # Display the graph in Streamlit
        st.pyplot(fig)
    
with tab2:
    st.header("Feature Importances")
    feat_imp_fig, ax1 = plt.subplots(figsize=(8, 6))  # Adjust the figsize as needed
    if selected_model_type == ModelType.LINEAR_REGRESSION.name:
        graphLinearRegression(model, selected_model_type, ax1, feat_imp_fig)
    elif selected_model_type == ModelType.GRADIANT_BOOSTING_REGRESSOR.name:
        graphGradientBoostingRegressor(model, ax1, feat_imp_fig)
    else:
        graphLinearModels(model, selected_model_type, ax1, feat_imp_fig)

with tab3:
    st.header("Classification Report")
    # Convert continuous predictions to class labels
    Y_test_preds_labels = (Y_test_preds > 0.5).astype(int)  # Assuming binary classification

    report = classification_report(Y_test, Y_test_preds_labels)
    st.text(report)

with tab4:
    sliders = []
    tab4_col1, tab4_col2 = st.columns(2)
    with tab4_col1:
        graphFeatureImportance(model, sliders)

    with tab4_col2:
        tab4_col2_col1, tab4_col2_col2 = st.columns(2, gap="medium")
        
        prediction = model.predict([sliders])
        popularity = prediction[0]  # Get the predicted popularity value
        
        with tab4_col2_col1:
            print(popularity)
            st.markdown("### Model Prediction: <strong style='color:tomato;'>{:d}</strong>".format(int(popularity)), unsafe_allow_html=True)
        
        with tab4_col2_col2:
            if selected_model_type == ModelType.RANDOM_FOREST_CLASSIFIER.name:
                graphRandomForestClassifierConfidence(model, sliders)
            else:
                st.metric(label="Model Confidence", value="NaN")