import numpy as np
import logging
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Create a logger
logger = logging.getLogger("evaluation")

TOLERANCE = 5

def evaluateModel(model, x, y):
    sklearnEvaluations(model, x, y)
    accuracyEvaluation(model, x, y)

def accuracyEvaluation(model, x, y):
    y_pred = model.predict(x)

    # Calculate the number of predictions within the tolerance level of the true values
    correct_predictions = np.abs(y - y_pred) <= TOLERANCE

    # Calculate the percentage of correct predictions
    correct_percentage = np.mean(correct_predictions) * 100

    logger.info(f'Accuracy (tolerance: {TOLERANCE}): {correct_percentage:.2f}%')

def sklearnEvaluations(model, x_test, y_test):
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"Mean Squared Error (MSE): {mse}")
    logger.info(f"Mean Absolute Error (MAE): {mae}")
    logger.info(f"R-squared (R²): {r2}")