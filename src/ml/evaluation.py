import json
import os
import numpy as np
import logging
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, f1_score
from prettytable import PrettyTable
from src.common.fileManagment import EVALUATION_RESULTS_DIRECTORY, getVersions

# Create a logger
logger = logging.getLogger("evaluation")

TOLERANCE = 0.1

def evaluateModel(model, x, y):
    evaluation_metrics = {}
  
    y_pred = model.predict(x)
    sklearn_metrics = sklearnEvaluations(y_pred, y, x)
    accuracy_metric = accuracyEvaluation(y_pred, y)

    evaluation_metrics.update(sklearn_metrics)
    evaluation_metrics.update(accuracy_metric)

    return evaluation_metrics, y_pred

def accuracyEvaluation(y_pred, y):
    # Calculate the number of predictions within the tolerance level of the true values
    correct_predictions = np.abs(y - y_pred) <= TOLERANCE

    # Calculate the percentage of correct predictions
    correct_percentage = np.mean(correct_predictions) * 100

    logger.info(f'Accuracy: {correct_percentage:.2f}%')

    return {'Accuracy': correct_percentage}

def sklearnEvaluations(y_pred, y_test, x_test):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    n = len(y_test)  # Number of samples
    p = x_test.shape[1]  # Number of features
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    n_bins = 10  # Number of bins (sqrt of total number of samples ==> popularity 0 - 100)
    bins = np.linspace(y_test.min(), y_test.max(), n_bins + 1)

    y_test_binned = np.digitize(y_test, bins)
    y_pred_binned = np.digitize(y_pred, bins)
    f1 = f1_score(y_test_binned, y_pred_binned, average= 'micro')

    logger.info(f"Mean Squared Error (MSE): {mse}, Root Mean Squared Error (RMSE): {rmse}, Mean Absolute Error (MAE): {mae}, R-squared (R²): {r2}, Adjusted R-squared: {adjusted_r2}, F1-Score: {f1}")

    return {'Mean Squared Error (MSE)': mse, 'Root Mean Squared Error (RMSE)': rmse, 'Mean Absolute Error (MAE)': mae, 'R-squared (R²)': r2, 'Adjusted R-squared (aR²)': adjusted_r2, 'F-Statistic (F1)': f1}


def compareEvaluations(model_type, version1=None, version2=None):
    base_directory_training = 'trainedModels'
    
    if version1 is None and version2 is None:
        # If no versions are specified, find the two newest versions
        version_dirs = getVersions()
        
        if len(version_dirs) < 2:
            logger.info("Not enough versions available for comparison.")
            return
        
        version2, version1 = version_dirs[:2]  # Assign version2 to the newer version and version1 to the older version
    
    results_filename = f"{model_type}_evaluation.json"
    metadata_filename = "metadata.json"
    
    full_path1 = os.path.join(EVALUATION_RESULTS_DIRECTORY, version1, results_filename)
    full_path2 = os.path.join(EVALUATION_RESULTS_DIRECTORY, version2, results_filename)
    metadata_path1 = os.path.join(base_directory_training, version1, metadata_filename)
    metadata_path2 = os.path.join(base_directory_training, version2, metadata_filename)
    
    if not os.path.exists(full_path1) or not os.path.exists(full_path2):
        logger.info("Evaluation results not found for one or both versions.")
        return
    
    with open(full_path1, 'r') as file1, open(full_path2, 'r') as file2:
        results1 = json.load(file1)
        results2 = json.load(file2)
    
    if os.path.exists(metadata_path1) and os.path.exists(metadata_path2):
        with open(metadata_path1, 'r') as file1, open(metadata_path2, 'r') as file2:
            metadata1 = json.load(file1)
            metadata2 = json.load(file2)
        logger.debug(f"Metadata for {version1}: {metadata1}")
        logger.debug(f"Metadata for {version2}: {metadata2}")
        dataset_length1 = metadata1.get('datasetLen', 'Unknown')
        dataset_length2 = metadata2.get('datasetLen', 'Unknown')
    else:
        dataset_length1 = 'Unknown'
        dataset_length2 = 'Unknown'
    
    table = PrettyTable()
    table.field_names = ["Metric", version1, version2, "Better Version"]
    
    for metric in results1:
        value1 = results1[metric]
        value2 = results2[metric]
        
        if metric in ['R-squared (R\u00b2)', 'Accuracy (tolerance: 5)']: # Higher is better
            if value1 > value2:
                better_version = version1
            elif value2 > value1:
                better_version = version2
            else:
                better_version = "Equal"
        else: # Lower is better
            if value1 < value2:
                better_version = version1
            elif value2 < value1:
                better_version = version2
            else:
                better_version = "Equal"
        
        table.add_row([metric, value1, value2, better_version])
    
    logger.info(f"Comparison between {version1} (Dataset Length: {dataset_length1}) and {version2} (Dataset Length: {dataset_length2}):")
    logger.info(table.get_string())  # Log the table as a string