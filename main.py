import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from pymongo import MongoClient
from db import add_song_if_not_exists
from spotify import getSongData

client_id = '98d7d7247b8e4cb3a1c7f6257ee1fa61'
client_secret = 'ab96f6f8fb9141edae374086459e3047'

client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Create a connection using MongoClient. This will connect to the default host and port.
client = MongoClient()

# Access database
spotifyDB = client['SpotifyRec']  # Replace 'mydatabase' with your database name

# Access collection of the database
songCollection = spotifyDB['RawSpotifySongs']  # Replace 'mycollection' with your collection name

# Get all available genre seeds
genres = sp.recommendation_genre_seeds()['genres']
# TODO remove this - only for testing
genres = [genres[0]]

# Get recommendations for each genre
for genre in genres:
    try:
        print(f"Fetching recommendations for genre: {genre}")
        recommendations = sp.recommendations(seed_genres=[genre], limit=100)
        songData = getSongData(recommendations, genre)
        print(songData)
    except Exception as e:
        print(f"Error fetching recommendations for genre {genre}: {e}")
    add_song_if_not_exists(songCollection, songData)

print("finish")
# # Example of calling an endpoint
# track_id = '2XpV9sHBexcNrz0Gyf3l18'
# features = sp.audio_features([track_id])
# print(features)
