from src.ml.evaluation import compareEvaluations
from src.ml.models import ModelType
import logging

MODEL_TYPES = [ModelType.LINEAR_REGRESSION, ModelType.RANDOM_FOREST_CLASSIFIER, ModelType.RANDOM_FOREST_REGRESSOR, ModelType.SVR]
#MODEL_TYPES = [ModelType.LINEAR_REGRESSION]
#MODEL_TYPES = []

# Configure logging 
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    handlers=[
                        logging.FileHandler('app.log', encoding='utf-8'),
                        logging.StreamHandler()
                    ])

# Create a logger
logger = logging.getLogger("comparison")


for modelType in MODEL_TYPES:
    logger.info(f"=========================Comparing {modelType.name}=========================")
    compareEvaluations(modelType.name)  # Compare the two newest versions