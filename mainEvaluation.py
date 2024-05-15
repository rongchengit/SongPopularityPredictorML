import pandas as pd
from sklearn.inspection import permutation_importance
from src.common.fileManagment import loadModel, saveEvaluation, saveNumpyFile, savePickleFile
from src.ml.evaluation import evaluateModel
from src.common.db import loadDBData
from src.ml.models import ModelType
from src.ml.sklearn import prepareData
import logging

MODEL_TYPES = list(ModelType)
#MODEL_TYPES = [ModelType.SVR]

# Configure logging 
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    handlers=[
                        logging.FileHandler('app.log', encoding='utf-8'),
                        logging.StreamHandler()
                    ])

# Create a logger
logger = logging.getLogger("main")

counter = 4
while True:
    counter = counter + 1
    version = f'v{counter}'
    songCollection = loadDBData(version)
    df = pd.DataFrame(songCollection)
    df, x_train, x_test, y_train, y_test, genre = prepareData(df, version)

    for modelType in MODEL_TYPES:
        logger.info(f"=========================Evaluating {modelType.name}=========================")
        model, version = loadModel(modelType.name, version)
        if model:
            # Evaluate against Training Datas
            evaluationMetricsTraining, y_pred_train = evaluateModel(model, x_train, y_train)
            saveEvaluation(evaluationMetricsTraining, modelType.name, 'evaluationTraining', version)
            saveNumpyFile(y_pred_train, modelType.name, 'predictionTraining', version)

            # Evaluate against Test Data
            evaluationMetrics, y_pred = evaluateModel(model, x_test, y_test)
            saveEvaluation(evaluationMetrics, modelType.name, 'evaluation', version)
            saveNumpyFile(y_pred, modelType.name, 'prediction', version)

            # SVR Feature Importance against Test Data
            if modelType == ModelType.SVR:
               model_named_steps = model.named_steps[ModelType.SVR.name.lower().replace("_", "")]
               result = permutation_importance(model_named_steps, model.named_steps['standardscaler'].transform(x_test), y_test, n_repeats=5, random_state=42, n_jobs=-1)
               savePickleFile(result, modelType.name, 'featureImportance', version)
        else:
            logger.error(f"Failed to load model {modelType.name}")


        
        