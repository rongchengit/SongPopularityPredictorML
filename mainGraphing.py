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
#make a map where the key is the attribute names for key and the values should be a object of min and max 
#so acoustics min:1 max:100 
# Prep Data
# songCollection = getCollection()
# data = list(songCollection.find())
# df = pd.DataFrame(data)
# x_train, x_test, y_train, y_test = prepareData(df)

#attributes = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence','duration_ms','key','loudness','mode','speechiness','tempo','time_signature']
# Load the audioRanges.npy file
audio_ranges_data = np.load('audioRanges.npy', allow_pickle=True).item()

# Create the audioFeatureRange dictionary using dictionary comprehension
audioFeatureRange = {attr: {'min': audio_ranges_data[attr][0], 'max': audio_ranges_data[attr][1]} for attr in audio_ranges_data}

Y_test = np.load('testData.npy')

@st.cache_resource
def load_model():
    model, version = loadModel(ModelType.RANDOM_FOREST_CLASSIFIER.name)
    return model

# Load the model
model = load_model()

# Make predictions
Y_test_preds = np.load('predictions.npy')

## Dashboard
st.title("PENIS :red[Prediction] :bar_chart: :chart_with_upwards_trend: :tea: :eggplant:")
st.markdown("Predict Wine Type using Ingredients Values")

tab1, tab2, tab3 = st.tabs(["important tab", "Penis indicator","penis graph"])

with tab1:
    st.header("Feature Importances")
    feat_imp_fig, ax1 = plt.subplots(figsize=(8, 6))  # Adjust the figsize as needed
    skplt.estimators.plot_feature_importances(model.named_steps['randomforestclassifier'], 
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
            if audioFeature == 'explicit':
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
                    label="Duration",
                    value=f'{(min_minutes + max_minutes) // 2:02d}:{(min_seconds + max_seconds) // 2:02d}'
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
            st.markdown("### Model Prediction: <strong style='color:tomato;'>{:.2f}</strong>".format(popularity), unsafe_allow_html=True)
        
        with col2:
            probs = model.predict_proba([sliders])
            probability = probs[0][1]  # Assuming the positive class is at index 1
            st.metric(label="Model Confidence", value="{:.2f} %".format(probability*100), delta="{:.2f} %".format((probability-0.5)*100))

        # explainer = lime_tabular.LimeTabularExplainer(x_train.values, mode="regression", feature_names=model.feature_names_in_)

        # # Reshape the sliders to a 2D array with shape (1, num_features)
        # instance = np.array(sliders).reshape(1, -1)

        # # Check if the number of features in the instance matches the expected number of features
        # if instance.shape[1] != len(model.feature_names_in_):
        #     raise ValueError(f"The number of features in the instance ({instance.shape[1]}) does not match the expected number of features ({len(model.feature_names_in_)}).")

        # # Ensure that the predict function returns a 1D array
        # prediction = model.predict(instance).flatten()

        # explanation = explainer.explain_instance(instance[0], model.predict, num_features=len(model.feature_names_in_))
        # interpretation_fig = explanation.as_pyplot_figure(label=prediction[0])
        # st.pyplot(interpretation_fig, use_container_width=True)