import streamlit as st

from src.ml.sklearn import loadModel
from src.ml.models import ModelType
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import logging
import matplotlib.pyplot as plt
import scikitplot as skplt
import numpy as np
import pandas as pd
from src.common.db import getCollection
from src.ml.sklearn import prepareData
from sklearn.inspection import permutation_importance #added for SVR
from lime import lime_tabular

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

# Load the audioRanges.npy file
audio_ranges_data = np.load('audioRanges.npy', allow_pickle=True).item()

# Create the audioFeatureRange dictionary using dictionary comprehension
audioFeatureRange = {attr: {'min': audio_ranges_data[attr][0], 'max': audio_ranges_data[attr][1]} for attr in audio_ranges_data}

Y_test = np.load('testData.npy')

@st.cache_resource
def load_model(selected_model_type):
    model, version = loadModel(selected_model_type.name)
    return model

# Make predictions
Y_test_preds = np.load('predictions.npy')

## Dashboard
st.title("PENIS :red[Prediction] :bar_chart: :chart_with_upwards_trend: :tea: :eggplant:")
st.markdown("Predict Wine Type using Ingredients Values")

selected_model_type = st.selectbox("Select Model Type", list(ModelType))
# Load the model
model = load_model(selected_model_type)

tab1, tab2, tab3 = st.tabs(["important tab", "Penis indicator","penis graph"])

with tab1:
    st.header("Feature Importances")
    feat_imp_fig, ax1 = plt.subplots(figsize=(8, 6))  # Adjust the figsize as needed
    if selected_model_type == ModelType.LINEAR_REGRESSION:
        # Get the linear classifier model from the pipeline
        model_named_steps = model.named_steps[selected_model_type.name.lower().replace("_", "")]
        # Get the coefficients of the linear classifier
        coefficients = model_named_steps.coef_

        # Check if coefficients is a single value or an array
        if isinstance(coefficients, np.float64):
            # If it's a single value, convert it to a list
            coefficients = [coefficients]
        else:
            # If it's an array, flatten it to a 1D list
            coefficients = coefficients.flatten().tolist()

        # Calculate the absolute values of the coefficients
        abs_coefficients = [abs(coef) for coef in coefficients]

        # Normalize the absolute coefficients to get the feature importances
        feature_importances = [coef / sum(abs_coefficients) for coef in abs_coefficients]

        # Sort the feature importances and feature names in descending order
        sorted_indices = np.argsort(feature_importances)[::-1]
        sorted_importances = [feature_importances[i] for i in sorted_indices]
        sorted_feature_names = [model.feature_names_in_[i] for i in sorted_indices]

        # Create a bar plot of the feature importances
        ax1.bar(range(len(sorted_importances)), sorted_importances)

        # Set the x-tick labels to the feature names
        ax1.set_xticks(range(len(sorted_importances)))
        ax1.set_xticklabels(sorted_feature_names, rotation=70)

        # Set the plot title and labels
        ax1.set_title("Feature Importances")
        ax1.set_xlabel("Features")
        ax1.set_ylabel("Importance")

        st.pyplot(feat_imp_fig, use_container_width=True)
    elif selected_model_type == ModelType.GRADIANT_BOOSTING_REGRESSOR:
        # Get the feature importances from the model
        feature_importances = model.feature_importances_

        # Sort the feature importances and feature names in descending order
        sorted_indices = np.argsort(feature_importances)[::-1]
        sorted_importances = feature_importances[sorted_indices]
        sorted_feature_names = [model.feature_names_in_[i] for i in sorted_indices]

        # Create a bar plot of the sorted feature importances
        ax1.bar(range(len(sorted_importances)), sorted_importances)

        # Set the x-tick labels to the sorted feature names
        ax1.set_xticks(range(len(sorted_importances)))
        ax1.set_xticklabels(sorted_feature_names, rotation=70)

        # Set the plot title and labels
        ax1.set_title("Feature Importances")
        ax1.set_xlabel("Features")
        ax1.set_ylabel("Importance")

        st.pyplot(feat_imp_fig, use_container_width=True)
    else:
        # Get the linear classifier model from the pipeline
        model_named_steps = model.named_steps[selected_model_type.name.lower().replace("_", "")]
        skplt.estimators.plot_feature_importances(model_named_steps, 
                                                feature_names=model.feature_names_in_, 
                                                ax=ax1, x_tick_rotation=70)
        st.pyplot(feat_imp_fig, use_container_width=True)

with tab2:
    st.header("Classification Report")
    report = classification_report(Y_test, Y_test_preds)
    st.text(report)

with tab3:
    sliders = []
    col1, col2 = st.columns(2)
    with col1:
        for audioFeature in model.feature_names_in_:
            if audioFeature == 'explicit' or audioFeature == 'mode':
                # Boolean feature
                checkbox_value = st.checkbox(label=audioFeature)
                sliders.append(1 if checkbox_value else 0)
            elif audioFeature == 'year':
                min_year = int(audioFeatureRange['year']['min'])
                max_year = int(audioFeatureRange['year']['max'])
                year_value = st.number_input(label='Year', min_value=min_year, max_value=max_year, step=1, value=int((min_year + max_year) / 2))
                sliders.append(year_value)
            elif audioFeature == 'duration_ms':
                # Duration feature (converted to minutes and seconds)
                min_duration = int(audioFeatureRange['duration_ms']['min'] / 1000)  # Convert milliseconds to seconds
                max_duration = int(audioFeatureRange['duration_ms']['max'] / 1000)  # Convert milliseconds to seconds
                min_minutes, min_seconds = divmod(min_duration, 60)
                max_minutes, max_seconds = divmod(max_duration, 60)
                duration_str = st.text_input(
                    label="Song Duration",
                    value= '3:00'
                )
                if duration_str:
                    try:
                        duration_parts = duration_str.split(':')
                        duration_minutes = int(duration_parts[0])
                        duration_seconds = int(duration_parts[1])
                        if (duration_minutes < min_minutes or
                            (duration_minutes == min_minutes and duration_seconds < min_seconds) or
                            duration_minutes > max_minutes or
                            (duration_minutes == max_minutes and duration_seconds > max_seconds)):
                            st.warning('Please enter a duration within the valid range.')
                        else:
                            duration_ms = (duration_minutes * 60 + duration_seconds) * 1000  # Convert back to milliseconds
                            sliders.append(duration_ms)
                    except (ValueError, IndexError):
                        st.warning('Please enter the duration in the format MM:SS.')
                else:
                    st.warning('Please enter a duration.')
            else:
                # Numeric feature
                if audioFeature == 'key' or audioFeature == 'time_signature':
                    min_value = int(audioFeatureRange[audioFeature]['min'])
                    max_value = int(audioFeatureRange[audioFeature]['max'])
                    median_value = round((min_value + max_value) / 2)
                    ing_slider = st.slider(label=audioFeature, min_value=min_value, max_value=max_value, value=median_value, step=1)
                else:
                    min_value = float(audioFeatureRange[audioFeature]['min'])
                    max_value = float(audioFeatureRange[audioFeature]['max'])
                    median_value = (min_value + max_value) / 2
                    ing_slider = st.slider(label=audioFeature, min_value=min_value, max_value=max_value, value=median_value)
                sliders.append(ing_slider)    

    with col2:
        col1, col2 = st.columns(2, gap="medium")
        
        prediction = model.predict([sliders])
        popularity = prediction[0]  # Get the predicted popularity value
        
        with col1:
            print(popularity)
            st.markdown("### Model Prediction: <strong style='color:tomato;'>{:d}</strong>".format(int(popularity)), unsafe_allow_html=True)
        
        with col2:
            if selected_model_type == ModelType.RANDOM_FOREST_CLASSIFIER:
                probs = model.predict_proba([sliders])
                probability = probs[0][1]  # Assuming the positive class is at index 1
                st.metric(label="Model Confidence", value="{:.2f} %".format(probability*100), delta="{:.2f} %".format((probability-0.5)*100))
            else:
                st.metric(label="Model Confidence", value="NaN")