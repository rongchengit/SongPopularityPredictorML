import streamlit as st
from src.common.fileManagment import loadEvaluations, getVersions, loadNumpyFile, loadPickleFile, loadModelFromS3, loadModel
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
import warnings
import os
logo_path = 'squid.png'

warnings.filterwarnings(action='ignore', category=UserWarning, message='X does not have valid feature names')
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

def get_model_type_options():
    if os.environ.get("CLOUD"):
        return [member.name for member in ModelType if member != ModelType.RANDOM_FOREST_CLASSIFIER and member != ModelType.RANDOM_FOREST_REGRESSOR]
    else:
        return [member.name for member in ModelType]

@st.cache_resource
def load_model(selected_model_type, selected_version):
    if os.environ.get("CLOUD"):
        model, version = loadModelFromS3(selected_model_type, selected_version)
    else:
        model, version = loadModel(selected_model_type, selected_version)
    return model

@st.cache_resource
def load_data(selected_version):
    songCollection = loadDBData(selected_version)
    df = pd.DataFrame(songCollection)
    df_with_track_id = df.copy().sort_values("popularity", ascending=False)
    df_with_track_id, _, _, _, _, _ = prepareData(df_with_track_id, selected_version, False)
    df, x_train, x_test, y_train, y_test, genre_mapping = prepareData(df, selected_version, True)
    return df, df_with_track_id, x_train, x_test, y_train, y_test, genre_mapping

## Dashboard
st.set_page_config(layout="wide")
col1 = st.columns([2, 1, 9.2])
with col1[0]:
    st.title(":green[Song Popularity ML]")

with col1[1]:
    st.text("")
    logo = st.image(logo_path, caption='', width=50)
st.markdown("Correlations between Song Popularity and Their Audio Features using MLs")

col2 = st.columns([2, 1, 7])
with col2[0]:
    selected_model_type = st.selectbox("Select Model Type", get_model_type_options()) # TODO make this only available in tab2 and tab4
with col2[1]:
    selected_version= st.selectbox("Select Version", getVersions())

tab1, tab2, tab3, tab4 = st.tabs(["Data Exploration", "Model Analysis/Fitting/Comparsions", "Fine Tuning/Predicting", "Post Evaluation and Song Recs"])

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
        # # with tab1Col3[1]:
        # #     generate_report(df, feature)
    
    tab1Col4 = st.columns(2)
    with tab1Col4[0]:
        with tab1Col2[1]:
            if genre != "ALL":
                encoded_genre = genre_mapping[genre]
                df = df[df['genre'] == encoded_genre]
            generate_scatterplot(df, feature)
    with tab1Col4[1]:
        correlations = df.corr(method='pearson')
        graphPopularityCorrelations(correlations)
    with tab1Col4[0]:
        generate_report(df, feature)
    #st.header("Classification Report")
    #report = classification_report(y_test, np.round(preds).astype(int), zero_division=1)
    #st.text(report)

with tab2:
    st.title("Model Analysis/Fitting/Comparsions")

    # Load Evals
    if selected_model_type != None:
        model = load_model(selected_model_type, selected_version)

    featureImportances = loadPickleFile('featureImportance', selected_version)

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

    tab2Col1 = st.columns(2)
    
    with tab2Col1[0]:
        if selected_model_type == ModelType.LINEAR_REGRESSION.name:
            graphLinearRegression(model, selected_model_type)
        elif selected_model_type == ModelType.GRADIENT_BOOSTING_REGRESSOR.name:
            graphGradientBoostingRegressor(model, selected_model_type)
        elif selected_model_type == ModelType.SVR.name:
            graphSVR(model, selected_model_type, featureImportances)
        else:
            graphLinearModels(model, selected_model_type)

    with tab2Col1[1]:
        graphModelFitting(model_names, mae_values, mae_values_training, 'Accuracy')

    tab2Col2 = st.columns(2)

    with tab2Col2[0]:
        graphModelFitting(model_names, mse_values, mse_values_training, 'MSE')
    with tab2Col2[1]:
        graphModelFitting(model_names, rmse_values, rmse_values_training, 'RMSE')
    
    tab2Col3 = st.columns(2)

    with tab2Col3[0]:
        graphModelFitting(model_names, mae_values, mae_values_training, 'MAE')
    with tab2Col3[1]:
        graphModelFitting(model_names, f1_values, f1_values_training, 'F1')

    tab2Col4 = st.columns(2)

    with tab2Col4[0]:
        graphModelFitting(model_names, mse_values, mse_values_training, 'R^2')
    with tab2Col4[1]:
        graphModelFitting(model_names, adjusted_r2_values, adjusted_r2_values_training, 'Adjusted R^2')

with tab3:
    sliders = {}
    tab3Col1 = st.columns([3,1])
    with tab3Col1[0]:
        audio_ranges_data = {col: (x_train.loc[x_train[col] != 0, col].min() if col == 'year' else x_train[col].min(), x_train[col].max()) for col in x_train.columns}
        audioFeatureRange = {attr: {'min': audio_ranges_data[attr][0], 'max': audio_ranges_data[attr][1]} for attr in audio_ranges_data}
        graphFeatureImportance(model, sliders, audioFeatureRange)

    with tab3Col1[1]:
        tab3Col2 = st.columns(2, gap="medium")

        prediction = model.predict([list(sliders.values())])
        popularity = prediction[0]  # Get the predicted popularity value
        
        with tab3Col2[0]:
            st.markdown("### Model Prediction: <strong style='color:green;'>{:d}</strong>".format(int(popularity)), unsafe_allow_html=True)
        
        with tab3Col2[1]:
            if selected_model_type == ModelType.RANDOM_FOREST_CLASSIFIER.name:
                graphRandomForestClassifierConfidence(model, list(sliders.values()))
            else:
                st.metric(label="Model Confidence", value="NaN")

# Create a custom filter function
def filter_track_ids(search_query, track_ids):
    if search_query:
        filtered_track_ids = [track_id for track_id in track_ids if search_query.lower() in str(track_id).lower()][:500]
    else:
        filtered_track_ids = track_ids[:500]
    return filtered_track_ids

with tab4:
    st.title("Post Evaluation and Song Recs")
    tab4Col1 = st.columns(2)
    df, df_with_track_id, x_train, x_test, y_train, y_test, genre_mapping = load_data(selected_version)

    # Input field for entering songId
    with tab4Col1[0]:
        search_query = st.text_input("Filter song IDs", key="searchQuery")
        genre = st.selectbox("Select Genre", ["ALL"] + list(genre_mapping.keys()))
        if genre != "ALL":
            encoded_genre = genre_mapping[genre]
            genre_df = df_with_track_id[df_with_track_id['genre'] == encoded_genre]
            filtered_track_ids = filter_track_ids(search_query, genre_df["track_id"])
        else:
            filtered_track_ids = filter_track_ids(search_query, df_with_track_id["track_id"])
            
        trackId = st.selectbox("Select a song ID", filtered_track_ids, key="trackIdInput")

    # Table to display songId, predicted popularity, and actual popularity
    with tab4Col1[1]:
        baseSong = {}
        recommendedSongs = []
        artists = []
        predictedPopularity = []
        actualPopularity = []
        if trackId:
            songs = pd.concat([recommend_songs(trackId, df_with_track_id), df_with_track_id[df_with_track_id["track_id"] == trackId]])
            songNames = getSongByTrackIds(songs["track_id"])

            for index, song in songs.iterrows():
                # Exclude the 'track_id' column from the song row
                numeric_features = song.drop('track_id').drop('popularity').values.tolist()
                # Reshape the numeric features into a 2D array
                numeric_features = np.array(numeric_features).reshape(1, -1)

                if song["track_id"] == trackId:
                    baseSong["songName"] = songNames[song["track_id"]]['songName']
                    baseSong["predictedPopularity"] = str(round(model.predict(numeric_features)[0]))
                    baseSong["actualPopularity"] = song["popularity"]
                    baseSong["artists"] = songNames[song["track_id"]]['artists']
                else:
                    recommendedSongs.append(str(songNames[song["track_id"]]['songName']))
                    predictedPopularity.append(str(round(model.predict(numeric_features)[0])))
                    actualPopularity.append(str(song["popularity"]))
                    artists.append(str(songNames[song["track_id"]]['artists']))

        data = {
            'Recommended Songs': recommendedSongs,
            'Artists': artists,
            'Predicted Popularity': predictedPopularity,
            'Actual Popularity': actualPopularity
        }
        songRecommendationDf = pd.DataFrame(data)

        st.text(f"Base Song: {baseSong['songName']} Name: ({baseSong['artists']}) Predicted Popularity: {baseSong['predictedPopularity']} Actual Popularity: {baseSong['actualPopularity']}")
        st.table(songRecommendationDf)