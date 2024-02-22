import pandas as pd
from src.ml.evaluation import evaluateModel
from src.common.db import getCollection
from src.ml.sklearn import prepareData, storeModel, trainModel, recommend_songs
import logging
from src.ml.models import ModelType

#MODEL_TYPES = [ModelType.LINEAR_REGRESSION, ModelType.RANDOM_FOREST_CLASSIFIER, ModelType.RANDOM_FOREST_REGRESSOR, ModelType.SVR]
#MODEL_TYPES = [ModelType.GRADIANT_BOOSTING_REGRESSOR]
MODEL_TYPES = []

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

# Convert into pandas dataframe
data = list(songCollection.find())
df = pd.DataFrame(data)

# Get Recommended Songs (has nothing to do with mainTraining)
logger.info(recommend_songs('6UFhNbE4sLRUoM52kC4Xl4', df))

# Get prapared Training Data
x_train, x_test, y_train, y_test = prepareData(df)

# Loop through all wanted models
for modelType in MODEL_TYPES:
    logger.info("=========================")
    model = trainModel(x_train, y_train, modelType)
    storeModel(model, modelType.name)
    evaluateModel(model, x_test, y_test)
    logger.info("=========================")