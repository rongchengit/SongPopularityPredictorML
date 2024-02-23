from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import joblib
import logging
import os
import pandas as pd

# Create a logger
logger = logging.getLogger("dataframe")

def prepareData(df):
    # Not needed for training
    df.drop('_id', axis=1, inplace=True)
    df.drop('track_id', axis=1, inplace=True)

    df.drop('artists', axis=1, inplace=True)

    #df[['year', 'month', 'day']] = df.apply(lambda row: extract_date_parts(row['release_date']), axis=1, result_type="expand")
    df['year'] = df.apply(lambda row: extract_date_parts(row['release_date'])[0], axis=1, result_type="expand")
    df.drop('release_date', axis=1, inplace=True)

    # Change genre to numerical
    le = LabelEncoder()
    df['genre'] = le.fit_transform(df['genre'])
    df.drop('genre', axis=1, inplace=True)
    # Log Dataframe for debugging
    logging.debug(f"\n{df.head().to_string()}")

    #X and Y
    x = df.drop(['popularity'], axis=1)
    y = df['popularity']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 
    return x_train, x_test, y_train, y_test

def trainModel(x, y, modelType):
    logger.info(f'Training {modelType.name} Model')
    model = make_pipeline(StandardScaler(), modelType.value)
    model.fit(x, y)
    return model

def storeModel(model, name):
    directory = 'trainedModels'
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)
    model_filename = name + '.joblib'
    full_path = os.path.join(directory, model_filename)
    joblib.dump(model, full_path)
    logger.debug(f'Model saved to {full_path}')
   
def recommend_songs(track_id, df, n_recommendations=5):

    # Example weights (make sure to align these with your actual feature columns)
    weights = {
        'loudness': 2.0,
        'tempo': 2.0,
        'acousticness': 2.0,
    }

    # Assuming `df` is your DataFrame containing the feature columns
    for feature, weight in weights.items():
        df[feature] *= weight
    
    # Filter out the row with the given track_id
    input_features = df[df['track_id'] == track_id].drop(['track_id'], axis=1)

    # Make sure input_features is numeric and in the correct format for cosine_similarity
    if not input_features.empty:
        input_features = input_features.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Compute similarities
        df_numeric = df.drop(['track_id'], axis=1).apply(pd.to_numeric, errors='coerce').fillna(0)
        similarities = cosine_similarity(input_features, df_numeric)
        
        # Get the top n recommendations; -2 because the first result is the song itself
        recommended_indices = similarities[0].argsort()[-(n_recommendations+2):-1][::-1]
        
        # Return the recommended track_ids
        recommended_track_ids = df.iloc[recommended_indices]['track_id']
        return recommended_track_ids
    else:
        print(f"No track found with ID: {track_id}")
        return []
    
# Normalize date function to ensure it's in 'YYYY-MM-DD' format, ignoring time
def extract_date_parts(date_str):
    parts = date_str.split('-')
    year = month = day = None  # Default values

    if len(parts) == 3:  # Year-Month-Day
        year, month, day = parts
    elif len(parts) == 2:  # Year-Month or Year-Day
        year, month = parts
        day = 1  # Default day if missing
    else:  # Year or other formats
        year = parts[0]
        month = 1  # Default month if missing
        day = 1  # Default day if missing

    return int(year), int(month), int(day)