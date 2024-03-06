import pandas as pd
from src.ml.evaluation import evaluateModel
from src.common.db import getCollection
from src.ml.models import ModelType
from src.ml.sklearn import saveEvaluation, loadModel, prepareData
import logging

MODEL_TYPES = [ModelType.LINEAR_REGRESSION, ModelType.RANDOM_FOREST_CLASSIFIER, ModelType.RANDOM_FOREST_REGRESSOR, ModelType.SVR]
#MODEL_TYPES = [ModelType.LINEAR_REGRESSION]
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

#Prep Data
songCollection = getCollection()
data = list(songCollection.find())
df = pd.DataFrame(data)
x_train, x_test, y_train, y_test = prepareData(df)

for modelType in MODEL_TYPES:
    logger.info(f"=========================Evaluating {modelType.name}=========================")
    model = loadModel(modelType.name)
    if model:
        evaluationMetrics = evaluateModel(model, x_test, y_test)
        saveEvaluation(evaluationMetrics, modelType.name)
    else:
        logger.error(f"Failed to load model {modelType.name}")