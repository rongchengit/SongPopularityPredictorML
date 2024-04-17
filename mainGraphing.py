import streamlit as st
from src.common.fileManagment import loadEvaluations, loadModel, getVersions, loadNumpyFile, loadPickleFile
from src.ml.sklearn import prepareData, recommend_songs
from src.ml.models import ModelType
from src.gathering.spotify import getSongByTrackIds
from src.graphing.datavis import generate_plot, graphPopularityCorrelations, generate_report, generate_scatterplot
from src.graphing.modelPredictions import graphFeatureImportance, graphRandomForestClassifierConfidence
from src.graphing.featureImportances import graphLinearRegression, graphGradientBoostingRegressor, graphLinearModels, graphLinearModels, graphSVR
from src.graphing.modelComparison import graphModelFitting
from src.common.db import loadDBData
import pandas as pd
import logging
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
    df_with_track_id = df.copy()
    df_with_track_id, _, _, _, _, _ = prepareData(df_with_track_id, selected_version, False)
    df, x_train, x_test, y_train, y_test, genre_mapping = prepareData(df, selected_version, True)
    return df, df_with_track_id, x_train, x_test, y_train, y_test, genre_mapping

## Dashboard
st.set_page_config(layout="wide")
st.title(":green[Bublify] :green[:squid:] :minidisc: :eggplant:")
st.markdown("Correlations between Song Popularity and Their Audio Features using MLs")

col = st.columns([2, 1, 7])
with col[0]:
    selected_model_type = st.selectbox("Select Model Type", [member.name for member in ModelType]) # TODO make this only available in tab2 and tab4
with col[1]:
    selected_version= st.selectbox("Select Version", getVersions())

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Exploration", "Feature Importances", "Model Analysis/Fitting/Comparsions", "Fine Tuning/Predicting", "Post Evaluation and Song Recs"])

with tab1:
    st.title("Data Visualization")
    tab1Col1 = st.columns([2, 2, 6])
    df, df_with_track_id, x_train, x_test, y_train, y_test, genre_mapping = load_data(selected_version)
    with tab1Col1[0]:
        genre = st.selectbox("Select Genre", list(genre_mapping.keys()) + ["ALL"])
        if genre != "ALL":
            encoded_genre = genre_mapping[genre]
            df = df[df['genre'] == encoded_genre]
    
    with tab1Col1[1]:
        feature = st.selectbox("Select Feature", df.columns, index=df.columns.tolist().index("energy"))
        
    tab1Col2 = st.columns(2)
    with tab1Col2[0]:
        tab1Col3 = st.columns(2)
        with tab1Col3[0]:
            generate_plot(df, feature)
        with tab1Col3[1]:
            generate_report(df, feature)
    with tab1Col2[1]:
        if genre != "ALL":
            encoded_genre = genre_mapping[genre]
            df = df[df['genre'] == encoded_genre]
        generate_scatterplot(df, feature)

    correlations = df.corr(method='pearson')
    graphPopularityCorrelations(correlations)

    #st.header("Classification Report")
    #report = classification_report(y_test, np.round(preds).astype(int), zero_division=1)
    #st.text(report)

with tab2:
    st.header("Feature Importances")

    # Load the model
    if selected_model_type != None:
        model = load_model(selected_model_type, selected_version)

    featureImportances = loadPickleFile('featureImportance', selected_version)

    if selected_model_type == ModelType.LINEAR_REGRESSION.name:
        graphLinearRegression(model, selected_model_type)
    elif selected_model_type == ModelType.GRADIENT_BOOSTING_REGRESSOR.name:
        graphGradientBoostingRegressor(model, selected_model_type)
    elif selected_model_type == ModelType.SVR.name:
        graphSVR(model, selected_model_type, featureImportances)
    else:
        graphLinearModels(model, selected_model_type)

with tab3:
    st.title("Model Analysis/Fitting/Comparsions")

    # Load Evals
    evals = loadEvaluations('evaluation', selected_version)
    evalsTraining = loadEvaluations('evaluationTraining', selected_version)
    preds = loadNumpyFile('prediction', selected_version, selected_model_type)

    # Evaluations for models against test data
    model_names = [eval["name"] for eval in evals]
    mse_values = [eval["Mean Squared Error (MSE)"] for eval in evals]
    rmse_values = [eval["Root Mean Squared Error (RMSE)"] for eval in evals]
    mae_values = [eval["Mean Absolute Error (MAE)"] for eval in evals]
    r2_values = [eval["R-squared (R\u00b2)"] for eval in evals]
    adjusted_r2_values = [eval["Adjusted R-squared (aR\u00b2)"] for eval in evals]
    accuracy_values = [eval["Accuracy"] for eval in evals]
    f1_values = [eval["F-Statistic (F1)"] for eval in evals]

    # Evaluations for models against training data
    model_names_training = [eval["name"] for eval in evalsTraining]
    mse_values_training = [eval["Mean Squared Error (MSE)"] for eval in evalsTraining]
    rmse_values_training = [eval["Root Mean Squared Error (RMSE)"] for eval in evalsTraining]
    mae_values_training = [eval["Mean Absolute Error (MAE)"] for eval in evalsTraining]
    r2_values_training = [eval["R-squared (R\u00b2)"] for eval in evalsTraining]
    adjusted_r2_values_training = [eval["Adjusted R-squared (aR\u00b2)"] for eval in evalsTraining]
    accuracy_values_training = [eval["Accuracy"] for eval in evalsTraining]
    f1_values_training = [eval["F-Statistic (F1)"] for eval in evalsTraining]

    tab3Col1 = st.columns(2)
 
    with tab3Col1[0]:
        graphModelFitting(model_names, mse_values, mse_values_training, 'MSE')
    with tab3Col1[1]:
        graphModelFitting(model_names, rmse_values, rmse_values_training, 'RMSE')
    
    tab3Col2 = st.columns(2)
    with tab3Col2[0]:
        graphModelFitting(model_names, mae_values, mae_values_training, 'MAE')
    with tab3Col2[1]:
        graphModelFitting(model_names, f1_values, f1_values_training, 'F1')

    tab3Col3 = st.columns(2)
    with tab3Col3[0]:
        graphModelFitting(model_names, mse_values, mse_values_training, 'R^2')
    with tab3Col3[1]:
        graphModelFitting(model_names, adjusted_r2_values, adjusted_r2_values_training, 'Adjusted R^2')

    tab3Col4 = st.columns(2)
    with tab3Col4[0]:
        graphModelFitting(model_names, mae_values, mae_values_training, 'Accuracy')
    with tab3Col4[1]:
        #graphModelFitting(model_names, accuracy_values, accuracy_values_training, 'Accuracy')
        st.title("TODO Best Models in order in a graph")

with tab4:
    sliders = {}
    tab4Col1 = st.columns([3,1])
    with tab4Col1[0]:
        audio_ranges_data = {col: (x_train.loc[x_train[col] != 0, col].min() if col == 'year' else x_train[col].min(), x_train[col].max()) for col in x_train.columns}
        audioFeatureRange = {attr: {'min': audio_ranges_data[attr][0], 'max': audio_ranges_data[attr][1]} for attr in audio_ranges_data}
        graphFeatureImportance(model, sliders, audioFeatureRange)

    with tab4Col1[1]:
        tab4Col2 = st.columns(2, gap="medium")
        prediction = model.predict([list(sliders.values())])
        popularity = prediction[0]  # Get the predicted popularity value
        
        with tab4Col2[0]:
            st.markdown("### Model Prediction: <strong style='color:green;'>{:d}</strong>".format(int(popularity)), unsafe_allow_html=True)
        
        with tab4Col2[1]:
            if selected_model_type == ModelType.RANDOM_FOREST_CLASSIFIER.name:
                graphRandomForestClassifierConfidence(model, list(sliders.values()))
            else:
                st.metric(label="Model Confidence", value="NaN")

with tab5:
    st.title("Post Evaluation and Song Recs")
    tab5Col1 = st.columns(2)
    df, df_with_track_id, x_train, x_test, y_train, y_test, genre_mapping = load_data(selected_version)

    # Input field for entering songId
    with tab5Col1[0]:
        trackId = st.text_input("Enter songId")

    # Table to display songId, predicted popularity, and actual popularity
    with tab5Col1[1]:
        recommendedSongs = []
        predictedPopularity = []
        actualPopularity = []
        if trackId:
            songs = recommend_songs(trackId, df_with_track_id)
            songNames = getSongByTrackIds(songs["track_id"])
            for index, song in songs.iterrows():
                # Exclude the 'track_id' column from the song row
                numeric_features = song.drop('track_id').drop('popularity').values.tolist()
                # Reshape the numeric features into a 2D array
                numeric_features = np.array(numeric_features).reshape(1, -1)

                recommendedSongs.append(songNames[song["track_id"]])
                predictedPopularity.append(round(model.predict(numeric_features)[0]))
                actualPopularity.append(song["popularity"])

        data = {
            'Recommended Songs': recommendedSongs,
            'Predicted Popularity': predictedPopularity,
            'Actual Popularity': actualPopularity
        }
        songRecommendationDf = pd.DataFrame(data)
        st.table(songRecommendationDf)