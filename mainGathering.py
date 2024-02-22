from src.common.db import add_song_if_not_exists, getCollection
from src.gathering.spotify import addAudioFeatures, addRecommendations, getGenres
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    handlers=[
                        logging.FileHandler("app.log"),
                        logging.StreamHandler()
                    ])

# Create a logger
logger = logging.getLogger("main")

# Get Database
songCollection = getCollection()

# Get all available genre seeds
genres = getGenres()

# Store new Songs
song_list = []

# Get recommendations for each genre
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

    # Store Songs    
    add_song_if_not_exists(songCollection, song_list)

    # Reset List
    song_list = []

logger.info("finish")
