import pandas as pd
from src.common.fileManagment import saveModel, getNewVersion
from src.common.db import getCollection
from src.ml.sklearn import prepareData, trainModel
import logging
from src.ml.models import ModelType

MODEL_TYPES = list(ModelType)
#MODEL_TYPES = [ModelType.SVR]

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

#Data Prep
# Get Database
songCollection = getCollection()
# Convert into pandas dataframe
data = list(songCollection.find())

while True:
    df = pd.DataFrame(data)
    # Get prapared Training Data
    df, x_train, x_test, y_train, y_test, genre = prepareData(df)
    version = getNewVersion()

    # Loop through all wanted models
    for modelType in MODEL_TYPES:
        logger.info("=========================")
        model = trainModel(x_train, y_train, modelType)
        version = saveModel(model, modelType.name, len(x_train) + len(x_test), version)
