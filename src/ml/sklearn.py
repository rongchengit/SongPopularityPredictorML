from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from src.common.fileManagment import getNewVersion
import logging
import pandas as pd

# Create a logger
logger = logging.getLogger("dataframe")

def prepareData(df, selected_version=None, dropId=True):
    if not selected_version:
        selected_version = getNewVersion()
    
    # Drop unused Data
    df.drop('_id', axis=1, inplace=True)
    if dropId == True:
        df.drop('track_id', axis=1, inplace=True)

    # Transform Artists
    le_artists = LabelEncoder()
    df['artists'] = df['artists'].apply(lambda x: ','.join(x))
    df['artists'] = le_artists.fit_transform(df['artists'])

    # Transform Release Date
    df['year'] = df.apply(lambda row: extract_date_parts(row['release_date'])[0], axis=1, result_type="expand")
    df.drop('release_date', axis=1, inplace=True)

    # Transform Genre
    le = LabelEncoder()
    df['genre'] = le.fit_transform(df['genre'])
    genre_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    # Logging
    logging.debug(f"\n{df.head().to_string()}")

    if selected_version == 'v1':
        x = df.drop(['popularity'], axis=1)[:100000]
        y = df['popularity'][:100000]
        
    elif selected_version == 'v2':
        x = df.drop(['popularity'], axis=1)[:200000]
        y = df['popularity'][:200000]
    
    elif selected_version == 'v3':
        x = df.drop(['popularity'], axis=1)[:300000]
        y = df['popularity'][:300000]
    
    elif selected_version == 'v4':
        x = df.drop(['popularity'], axis=1)[:400000]
        y = df['popularity'][:400000]

    elif selected_version == 'v5':
        x = df.drop(['popularity'], axis=1)
        y = df['popularity']

    elif selected_version == 'v6': # ==> getting rid of outliers/reasonings
        Q1 = df['duration_ms'].quantile(0.25)
        Q3 = df['duration_ms'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define the outlier threshold
        outlier_threshold = 1.5

        # Filter out the outliers
        df = df[~((df['duration_ms'] < (Q1 - outlier_threshold * IQR)) | (df['duration_ms'] > (Q3 + outlier_threshold * IQR)))]
        x = df.drop(['popularity'], axis=1)
        y = df['popularity']
    
    elif selected_version == 'v7': # ==> specialized df by genre
        df = df[df['genre'] == 'anime'] 
        x = df.drop(['popularity'], axis=1)
        y = df['popularity']
        
    else:
        raise Exception("Version not yet implemented") 

    logging.debug(f"Preparing {selected_version} with Dataset Len: {len(x)}")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return df, x_train, x_test, y_train, y_test, genre_mapping

def trainModel(x, y, modelType):
    logger.info(f'Training {modelType.name} Model')
    model = make_pipeline(StandardScaler(), modelType.value)
    model.fit(x, y)
    return model

def recommend_songs(track_id, df, n_recommendations=5):
    internalDf = df.copy()
    # Example weights (make sure to align these with your actual feature columns)
    weights = {
        'genre': 4,
        'artists': 6,
        'popularity': 3
    }

    # Assuming `df` is your DataFrame containing the feature columns
    for feature, weight in weights.items():
        internalDf[feature] *= weight
    
    # Filter out the row with the given track_id
    input_features = internalDf[internalDf['track_id'] == track_id].drop(['track_id'], axis=1)

    # Make sure input_features is numeric and in the correct format for cosine_similarity
    if not input_features.empty:
        input_features = input_features.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Compute similarities
        df_numeric = internalDf.drop(['track_id'], axis=1).apply(pd.to_numeric, errors='coerce').fillna(0)
        similarities = cosine_similarity(input_features, df_numeric)
        
        # Get the top n recommendations; -2 because the first result is the song itself
        recommended_indices = similarities[0].argsort()[-(n_recommendations+2):-1][::-1]
        
        # Return the recommended tracks
        recommended_tracks = df.iloc[recommended_indices].squeeze()
        return recommended_tracks
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