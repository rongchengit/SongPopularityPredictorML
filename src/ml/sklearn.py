from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import joblib
import logging
import os
import pandas as pd
import json

# Create a logger
logger = logging.getLogger("dataframe")

def prepareData(df):
       # Not needed for training
    df.drop('_id', axis=1, inplace=True)
    df.drop('track_id', axis=1, inplace=True)

    le_artists = LabelEncoder()
    df['artists'] = df['artists'].apply(lambda x: ','.join(x))  # Convert artist array to comma-separated string
    df['artists'] = le_artists.fit_transform(df['artists'])

    df['year'] = df.apply(lambda row: extract_date_parts(row['release_date'])[0], axis=1, result_type="expand")
    df.drop('release_date', axis=1, inplace=True)

    #---------------------------------------------------------------- removing outliers from duration
    Q1 = df['duration_ms'].quantile(0.25)
    Q3 = df['duration_ms'].quantile(0.75)
    IQR = Q3 - Q1

    # Define the outlier threshold
    outlier_threshold = 1.5

    # Filter out the outliers
    df = df[~((df['duration_ms'] < (Q1 - outlier_threshold * IQR)) | (df['duration_ms'] > (Q3 + outlier_threshold * IQR)))]
    #------------------------------------------------------------------
    # Change genre to numerical
    le = LabelEncoder()
    df['genre'] = le.fit_transform(df['genre'])

    # Create a dictionary to map genre names to their encoded values
    genre_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    # Log Dataframe for debugging
    logging.debug(f"\n{df.head().to_string()}")

    #X and Y
    x = df.drop(['popularity'], axis=1)
    y = df['popularity']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 
    return df, x_train, x_test, y_train, y_test, genre_mapping

def trainModel(x, y, modelType):
    logger.info(f'Training {modelType.name} Model')
    model = make_pipeline(StandardScaler(), modelType.value)
    model.fit(x, y)
    return model

def storeModel(model, name, dataset_length, version=None):
    base_directory = 'trainedModels'
    version_directory, version = getVersionDirectory(base_directory, version)
    
    model_filename = name + '.joblib'
    full_path = os.path.join(version_directory, model_filename)
    joblib.dump(model, full_path)
    
    metadata_filename = 'metadata.json'
    metadata_path = os.path.join(version_directory, metadata_filename)
    
    if not os.path.exists(metadata_path):
        metadata = {'Dataset Length': dataset_length}
        with open(metadata_path, 'w') as file:
            json.dump(metadata, file, indent=4)
        logger.debug(f'Metadata saved to {metadata_path}')
    
    logger.debug(f'Model saved to {full_path}')
    
    return version

def saveEvaluation(evaluationMetrics, model_type, version=None):
    base_directory = 'evaluationResults'
    
    version_directory, version = getVersionDirectory(base_directory, version)
    
    results_filename = f"{model_type}_evaluation.json"
    full_path = os.path.join(version_directory, results_filename)
    
    with open(full_path, 'w') as file:
        json.dump(evaluationMetrics, file, indent=4)
    
    logger.debug(f'Evaluation results saved to {full_path}')
    return version

def getVersions():
    directory = 'trainedModels'
    versions = [d for d in os.listdir(directory) if d.startswith('v')]
    
    # Sort the versions in descending order
    sorted_versions = sorted(versions, key=lambda x: int(x[1:]), reverse=True)
    
    return sorted_versions

# Load the model via name (by default gets the newest model version)
def loadModel(name, version=None):
    directory = 'trainedModels'
    
    if version is None:
        # If no version is specified, find the latest version directory
        version_dirs = [d for d in os.listdir(directory) if d.startswith('v')]
        if not version_dirs:
            logger.error(f'No model versions found in {directory}.')
            return None, None
        version = max(version_dirs)
    
    version_directory = os.path.join(directory, version)
    model_filename = name + '.joblib'
    full_path = os.path.join(version_directory, model_filename)
    
    if os.path.exists(full_path):
        model = joblib.load(full_path)
        logger.info(f'Model loaded from {full_path}')
        return model, version
    else:
        logger.error(f'Model file {full_path} not found.')
        return None, None

def getVersionDirectory(base_directory, version=None):
    # Ensure the base directory exists
    os.makedirs(base_directory, exist_ok=True)
    
    if version is None:
        # Find the next available version directory
        version = 1
        while True:
            version_directory = os.path.join(base_directory, f'v{version}')
            if not os.path.exists(version_directory):
                break
            version += 1
    else:
        if isinstance(version, str) and version.startswith('v'):
            # Extract the numeric version from the string
            version = int(version[1:])
        
        version_directory = os.path.join(base_directory, f'v{version}')
    
    # Create the version directory if it doesn't exist
    os.makedirs(version_directory, exist_ok=True)
    
    return version_directory, f'v{version}'

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