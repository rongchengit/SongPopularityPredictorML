import streamlit as st

from src.ml.sklearn import loadModel
from src.ml.models import ModelType
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import logging
import matplotlib.pyplot as plt
import scikitplot as skplt
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

# Prep Data
Y_test = np.load('testData.npy')

## Load Model
model, version = loadModel(ModelType.RANDOM_FOREST_CLASSIFIER.name)

# Make predictions
Y_test_preds = np.load('predictions.npy')

## Dashboard
st.title("PENIS :red[Prediction] :bar_chart: :chart_with_upwards_trend: :tea: :coffee:")
st.markdown("Predict Wine Type using Ingredients Values")

tab1, = st.tabs(["Global Performance :weight_lifter:"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.header("Classification Report")
        report = classification_report(Y_test, Y_test_preds)
        st.text(report)

    with col2:
        st.header("Feature Importances")
        feat_imp_fig, ax1 = plt.subplots(figsize=(8, 6))  # Adjust the figsize as needed
        skplt.estimators.plot_feature_importances(model.named_steps['randomforestclassifier'], 
                                                feature_names=model.feature_names_in_, 
                                                ax=ax1, x_tick_rotation=70)
        st.pyplot(feat_imp_fig, use_container_width=True)