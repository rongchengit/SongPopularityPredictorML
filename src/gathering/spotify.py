import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import logging
import random

# Create a logger
logger = logging.getLogger("spotify")

# Spotify Web Credentials
client_id = 'a58b70de4d8b4024a8647572414d02b7'
client_secret = 'c13c36ea2749445f9e53897ce9ba0d84'

# Constants
TRACK_ID = "track_id"
attributes = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence','duration_ms','key','loudness','mode','speechiness','tempo','time_signature']
recommendationAttributes = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence', 'mode', 'speechiness']

client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager, retries=0, requests_timeout=5)

def getGenres():
    return sp.recommendation_genre_seeds()['genres']

def addRecommendations(song_list, genre):
    # Get randomized parameters for this iteration
    params = get_randomized_parameters()

    recommendations = sp.recommendations(seed_genres=[genre], limit=100, **params)
    for track in recommendations['tracks']:
        song_data = {
            TRACK_ID: track['id'],
            "duration_ms": track['duration_ms'],
            "popularity": track['popularity'],
            "genre": [genre]
        }
        song_list.append(song_data)

def addAudioFeatures(song_list):
    track_ids = [track[TRACK_ID] for track in song_list]

    # Split up audiofeatures request in 2 separate ones due to spotify answering with http code 414
    audio_features1 = sp.audio_features(track_ids[:50])
    audio_features2 = sp.audio_features(track_ids[50:100])
    audio_features = audio_features1 + audio_features2
    
    # Iterate over each song in songData
    for song in song_list:
        # Find matching audio features based on track ID
        matching_features = next((features for features in audio_features if features['id'] == song[TRACK_ID]), None)
        
        # If matching features found, add them to the song, excluding the specified keys
        if matching_features:
            for key, value in matching_features.items():
                if key in attributes:
                    song[key] = value

def get_randomized_parameters():
    # Generate random min, max, and target values for each attribute
    params = {}
    for attribute in recommendationAttributes:
        get_param(params, attribute, 0.0, 1.0, 1)
    
    # Other attributes
    get_param(params, 'key', 0, 11, 0)
    get_param(params, 'popularity', 0, 100, 0)
    
    return params

def get_param(params, attribute, min, max, nDigits):
    target_val = custom_round(random.uniform(min, max), nDigits)
    params[f'target_{attribute}'] = target_val

def custom_round(value, digits):
    if digits == 0:
        return int(round(value, digits))
    else:
        return round(value, digits)