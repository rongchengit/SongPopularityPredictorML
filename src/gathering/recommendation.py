import logging
import random
import time
from src.common.db import add_song_if_not_exists
from src.gathering.spotify import addAudioFeatures, addRecommendations, getGenres

# Create a logger
logger = logging.getLogger("recommendation")

def searchByRecommendation(songCollection):
    # Store new Songs
    song_list = []

    # Get recommendations for each genre
    while True: # Loop until we encounter the rate limit
        # Get all available genre seeds
        genres = getGenres()
        for genre in genres:
            try:
                logger.info(f"Fetching recommendations for genre: {genre}")

                # Fetch Song Recommendations
                while len(song_list) == 0:
                    addRecommendations(song_list, genre)
            
                # Fetch Audio Features for all track IDs
                addAudioFeatures(song_list)
            except Exception as e:
                logger.error(f"Error fetching recommendations for genre {genre}: {e}")
                return # Break out of the loop once an error (rate limit) occurs

            # Store Songs    
            add_song_if_not_exists(songCollection, song_list)

            # Reset List
            song_list = []
            
            time.sleep(random.uniform(5, 15)) 