import pandas as pd
from src.ml.evaluation import evaluateModel
from src.common.db import getCollection
from src.ml.models import ModelType
from src.ml.sklearn import saveEvaluation, loadModel, prepareData
import logging
import numpy as np

MODEL_TYPES = list(ModelType)
#MODEL_TYPES = [ModelType.SVR]

# Configure logging 
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    handlers=[
                        logging.FileHandler('app.log', encoding='utf-8'),
                        logging.StreamHandler()
                    ])

# Create a logger
logger = logging.getLogger("main")

#Prep Data
songCollection = getCollection()
data = list(songCollection.find())
df = pd.DataFrame(data)
df, x_train, x_test, y_train, y_test, genre = prepareData(df)

for modelType in MODEL_TYPES:
    logger.info(f"=========================Evaluating {modelType.name}=========================")
    model, version = loadModel(modelType.name)
    if model:
        evaluationMetrics, y_pred = evaluateModel(model, x_test, y_test)
        version = saveEvaluation(evaluationMetrics, modelType.name, version)
        audioFeatureRanges = {col: (x_train.loc[x_train[col] != 0, col].min() if col == 'year' else x_train[col].min(), x_train[col].max()) for col in x_train.columns}

        #if modelType == ModelType.SVR:
        #    model_named_steps = model.named_steps[ModelType.SVR.name.lower().replace("_", "")]
        #    result = permutation_importance(model_named_steps, model.named_steps['standardscaler'].transform(x_test), y_test, n_repeats=5, random_state=42)
        #    np.save('svrFeatureImportance.npy', result)

        np.save('predictions.npy', y_pred)
        np.save('testData.npy', y_test)
        np.save('audioRanges', audioFeatureRanges)
    else:
        logger.error(f"Failed to load model {modelType.name}")


        
        