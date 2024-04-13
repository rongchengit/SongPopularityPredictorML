import streamlit as st

from src.common.fileManagment import loadEvaluations, loadModel, getVersions, loadNumpyFile
from src.ml.sklearn import prepareData
from src.ml.models import ModelType
from src.graphing.datavis import generate_plot, generateBarChart, graphPopularityCorrelations
from src.graphing.modelPredictions import graphFeatureImportance, graphRandomForestClassifierConfidence
from src.graphing.featureImportances import graphLinearRegression, graphGradientBoostingRegressor, graphLinearModels, graphLinearModels, graphSVR
from sklearn.metrics import classification_report
from src.common.db import loadDBData
import pandas as pd
import logging
import matplotlib.pyplot as plt
import numpy as np

# Configure logging 
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    handlers=[
                        logging.FileHandler('app.log', encoding='utf-8'),
                        logging.StreamHandler()
                    ])

# Create a logger
logger = logging.getLogger("main")

@st.cache_resource
def load_model(selected_model_type, selected_version):
    model, version = loadModel(selected_model_type, selected_version)
    return model

@st.cache_resource
def load_data(selected_version):
    songCollection = loadDBData(selected_version)
    df = pd.DataFrame(songCollection)
    return prepareData(df, selected_version)

## Dashboard
st.set_page_config(layout="wide")
st.title(":green[Bublify] :green[:squid:] :minidisc: :eggplant:")
st.markdown("Correlations between Song Popularity and Their Audio Features using MLs")

# Create a sidebar for Filters
st.sidebar.title("Dropdown Filters")
selected_model_type = st.sidebar.selectbox("Select Model Type", [member.name for member in ModelType])
selected_version = st.sidebar.selectbox("Select Version", getVersions())

# Load Evals
evals = loadEvaluations('evaluation', selected_version)
evalsTraining = loadEvaluations('evaluationTraining', selected_version)
preds = loadNumpyFile('prediction', selected_version, selected_model_type)
featureImportances = loadNumpyFile('featureImportance', selected_version)

# Evaluations for models against test data
model_names = [eval["name"] for eval in evals]
mse_values = [eval["Mean Squared Error (MSE)"] for eval in evals]
mae_values = [eval["Mean Absolute Error (MAE)"] for eval in evals]

# Evaluations for models against training data
model_names_training = [eval["name"] for eval in evalsTraining]
mse_values_training = [eval["Mean Squared Error (MSE)"] for eval in evalsTraining]
mae_values_training = [eval["Mean Absolute Error (MAE)"] for eval in evalsTraining]

# Load the model
model = load_model(selected_model_type, selected_version)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Model Comparisons", "Feature Importances", "Classification Summaries", "Fine Tuning/Predicting", "Data Exploration", "Model Fitting"])

with tab1:
    st.title("Overview of Trained Models")
    tab1_col1, tab1_col2 = st.columns(2)
    with tab1_col1:
        # Sort the models based on MSE values in ascending order
        sorted_models_mse = sorted(zip(model_names, mse_values), key=lambda x: x[1])
        sorted_model_names_mse, sorted_mse_values = zip(*sorted_models_mse)
        generateBarChart(sorted_mse_values, sorted_model_names_mse, 'MSE')

    with tab1_col2:
        # Sort the models based on MAE values in ascending order
        sorted_models_mae= sorted(zip(model_names, mae_values), key=lambda x: x[1])
        sorted_model_names_mae, sorted_mae_values = zip(*sorted_models_mae)
        generateBarChart(sorted_mae_values, sorted_model_names_mae, 'MAE')
        
    st.subheader("Training Data")
    tab1_col3, tab1_col4 = st.columns(2)
    with tab1_col3:
        sorted_models_mse_training = sorted(zip(model_names_training, mse_values_training), key=lambda x: x[1])
        sorted_model_names_mse_training, sorted_mse_values_training = zip(*sorted_models_mse_training)
        generateBarChart(sorted_mse_values_training, sorted_model_names_mse_training, 'MSE')

    with tab1_col4:
        sorted_models_mae_training = sorted(zip(model_names_training, mae_values_training), key=lambda x: x[1])
        sorted_model_names_mae_training, sorted_mae_values_training = zip(*sorted_models_mae_training)
        generateBarChart(sorted_mae_values_training, sorted_model_names_mae_training, 'MAE')
        
with tab2:
    st.header("Feature Importances")
    tab2_col1, tab2_col2 = st.columns(2)

    with tab2_col1:
        feat_imp_fig, ax1 = plt.subplots(figsize=(8, 6))  # Adjust the figsize as needed
        if selected_model_type == ModelType.LINEAR_REGRESSION.name:
            graphLinearRegression(model, selected_model_type)
        elif selected_model_type == ModelType.GRADIENT_BOOSTING_REGRESSOR.name:
            graphGradientBoostingRegressor(model, selected_model_type)
        elif selected_model_type == ModelType.SVR.name:
            graphSVR(model, selected_model_type, featureImportances)
        else:
            graphLinearModels(model, selected_model_type)
    with tab2_col2:
        df, x_train, x_test, y_train, y_test, genre_mapping = load_data(selected_version)

        correlations = df.corr(method='pearson')
        graphPopularityCorrelations(correlations)

with tab3:
    st.header("Classification Report")
    df, x_train, x_test, y_train, y_test, genre_mapping = load_data(selected_version)
    report = classification_report(y_test, np.round(preds).astype(int), zero_division=1)
    st.text(report)

with tab4:
    sliders = {}
    tab4_col1, tab4_col2 = st.columns(2)
    with tab4_col1:
        audio_ranges_data = {col: (x_train.loc[x_train[col] != 0, col].min() if col == 'year' else x_train[col].min(), x_train[col].max()) for col in x_train.columns}
        audioFeatureRange = {attr: {'min': audio_ranges_data[attr][0], 'max': audio_ranges_data[attr][1]} for attr in audio_ranges_data}
        graphFeatureImportance(model, sliders, audioFeatureRange)

    with tab4_col2:
        tab4_col2_col1, tab4_col2_col2 = st.columns(2, gap="medium")
        prediction = model.predict([list(sliders.values())])
        popularity = prediction[0]  # Get the predicted popularity value
    
        
        with tab4_col2_col1:
            st.markdown("### Model Prediction: <strong style='color:green;'>{:d}</strong>".format(int(popularity)), unsafe_allow_html=True)
        
        with tab4_col2_col2:
            if selected_model_type == ModelType.RANDOM_FOREST_CLASSIFIER.name:
                graphRandomForestClassifierConfidence(model, list(sliders.values()))
            else:
                st.metric(label="Model Confidence", value="NaN")

with tab5:
    st.title("Outlier Visualization")
    tab5_col1, tab5_col2 = st.columns(2)
    df, x_train, x_test, y_train, y_test, genre_mapping = load_data(selected_version)
    with tab5_col1:
        genre = st.selectbox("Select Genre", list(genre_mapping.keys()))
        encoded_genre = genre_mapping[genre]
        x_combined = pd.concat([x_train[x_train['genre'] == encoded_genre], x_test[x_test['genre'] == encoded_genre]], axis=0)
    with tab5_col2:
         feature = st.selectbox("Select Feature", x_combined.columns, index=x_combined.columns.tolist().index("energy"))

    generate_plot(x_combined, feature)
