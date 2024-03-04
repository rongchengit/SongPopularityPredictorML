from src.gathering.recommendation import searchByRecommendation
from src.gathering.search import searchByTrackName
from src.common.db import getCollection
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
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

# Different methods to get random spread of songs
#searchByRecommendation(songCollection)
searchByTrackName(songCollection, 'death-metal')

logger.info("finish")
