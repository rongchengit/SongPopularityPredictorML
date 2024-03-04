import logging
import random
import time
from src.common.db import add_song_if_not_exists
from src.gathering.spotify import addAudioFeatures, addSearchedSongs, getGenres
import string

# Create a logger
logger = logging.getLogger("searchByTrack")

def searchByTrackName(songCollection, targetGenre=None):
    # Store new Songs
    song_list = []

    targetGenreReached = False if targetGenre else True

    # Get recommendations for each genre
    while True: # Loop until we encounter the rate limit
        # Get all available genre seeds
        genres = getGenres()
        for genre in genres:

             # Skip genres until the target is reached
            if not targetGenreReached:
                if genre == targetGenre:
                    targetGenreReached = True
                else:
                    continue

            # List of lowercase letters
            alphabet = string.ascii_lowercase
            two_letter_combos = [f"{first_letter}{second_letter}" 
                     for first_letter in alphabet 
                     for second_letter in alphabet]
            
            for letter in two_letter_combos:
                try:
                    logger.info(f"Fetching tracks {genre} and letter: {letter}")
                    
                    # Fetch Song Recommendations
                    addSearchedSongs(song_list, genre, letter)

                    if len(song_list) == 0:
                        break

                    # Fetch Audio Features for all track IDs
                    addAudioFeatures(song_list)
                except Exception as e:
                    logger.error(f"Error fetching songs for genre {genre} and letter {letter}: {e}")
                    return # Break out of the loop once an error (rate limit) occurs

                # Store Songs    
                add_song_if_not_exists(songCollection, song_list)

                # Reset List
                song_list = []
                
                time.sleep(random.uniform(5, 15)) 