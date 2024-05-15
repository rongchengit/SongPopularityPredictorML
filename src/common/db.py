from pymongo import MongoClient, ASCENDING
import logging
import os
from src.common.fileManagment import getVersions, loadMetadata

# Create a logger
logger = logging.getLogger("db")

def getCollection():
    if os.environ.get("ATLAS_PASSWORD"):
        client = MongoClient("mongodb+srv://rcrongc:" + os.environ.get("ATLAS_PASSWORD") + "@bublifymlcluster.ojxutzi.mongodb.net/?retryWrites=true&w=majority&appName=bublifyMLCluster")
    else:
        client = MongoClient()
    spotifyDB = client['SpotifyRec']
    return spotifyDB['AudioFeatures']

# Function to add a song if it doesn't exist in the database
def add_song_if_not_exists(collection, song_data):
    try: 
        for song in song_data:
            # Check if the song exists in the database
            if collection.find_one({"track_id": song["track_id"]}) is None:
                # Song doesn't exist, so add it
                collection.insert_one(song)
                logger.info(f"Song added: {song['track_id']}")
            else:
                # Song already exists
                logger.info(f"Song already exists: {song['track_id']}")
    except Exception as e:
        logger.error(f"Error Storing Song in Database: {e}")

def loadDBData(version: None):
    if version is None:
        version = getVersions()[-1]

    metadata = loadMetadata(version)

    songCollection = getCollection()
    if os.environ.get("CLOUD"):
        return list(songCollection.find().sort([("$natural", ASCENDING)]).limit(1000)) # Limit due to free tier hardware constraints
    else:
        return list(songCollection.find().sort([("$natural", ASCENDING)]).limit(metadata.get('datasetLen')))
    