import pandas as pd
from src.common.db import getCollection
from src.ml.sklearn import prepareData, storeModel, trainModel
import logging
from src.ml.models import ModelType

MODEL_TYPES = list(ModelType)
#MODEL_TYPES = []

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

#Data Prep
# Get Database
songCollection = getCollection()
# Convert into pandas dataframe
data = list(songCollection.find())
df = pd.DataFrame(data)

# Get prapared Training Data
x_train, x_test, y_train, y_test = prepareData(df)

# Get Recommended Songs (has nothing to do with mainTraining)
# logger.info(recommend_songs('6UFhNbE4sLRUoM52kC4Xl4', df))

version = None

# Loop through all wanted models
for modelType in MODEL_TYPES:
    logger.info("=========================")
    model = trainModel(x_train, y_train, modelType)
    version = storeModel(model, modelType.name, len(x_train), version)
