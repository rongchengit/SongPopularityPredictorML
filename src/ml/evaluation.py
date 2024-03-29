import json
import os
import numpy as np
import logging
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from prettytable import PrettyTable

# Create a logger
logger = logging.getLogger("evaluation")

BASE_DIRECTORY = 'evaluationResults'
TOLERANCE = 5

def evaluateModel(model, x, y):
    evaluation_metrics = {}
    y_pred = model.predict(x)
    sklearn_metrics = sklearnEvaluations(y_pred, y)
    accuracy_metric = accuracyEvaluation(y_pred, y)

    evaluation_metrics.update(sklearn_metrics)
    evaluation_metrics.update(accuracy_metric)

    return evaluation_metrics, y_pred

def accuracyEvaluation(y_pred, y):
    # Calculate the number of predictions within the tolerance level of the true values
    correct_predictions = np.abs(y - y_pred) <= TOLERANCE

    # Calculate the percentage of correct predictions
    correct_percentage = np.mean(correct_predictions) * 100

    logger.info(f'Accuracy (tolerance: {TOLERANCE}): {correct_percentage:.2f}%')

    return {'Accuracy (tolerance: {})'.format(TOLERANCE): correct_percentage}

def sklearnEvaluations(y_pred, y_test):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"Mean Squared Error (MSE): {mse}")
    logger.info(f"Mean Absolute Error (MAE): {mae}")
    logger.info(f"R-squared (R²): {r2}")

    return {'Mean Squared Error (MSE)': mse, 'Mean Absolute Error (MAE)': mae, 'R-squared (R²)': r2}


def compareEvaluations(model_type, version1=None, version2=None):
    base_directory_training = 'trainedModels'
    
    if version1 is None and version2 is None:
        # If no versions are specified, find the two newest versions
        version_dirs = [d for d in os.listdir(BASE_DIRECTORY) if d.startswith('v')]
        version_dirs.sort(reverse=True)
        
        if len(version_dirs) < 2:
            logger.info("Not enough versions available for comparison.")
            return
        
        version2, version1 = version_dirs[:2]  # Assign version2 to the newer version and version1 to the older version
    
    results_filename = f"{model_type}_evaluation.json"
    metadata_filename = "metadata.json"
    
    full_path1 = os.path.join(BASE_DIRECTORY, version1, results_filename)
    full_path2 = os.path.join(BASE_DIRECTORY, version2, results_filename)
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
        dataset_length1 = metadata1.get('Dataset Length', 'Unknown')
        dataset_length2 = metadata2.get('Dataset Length', 'Unknown')
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

def loadEvaluations(version):
    evaluations = []
    directory = os.path.join(BASE_DIRECTORY, version)
    # Iterate over the files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            
            # Load the JSON data from the file
            with open(file_path, "r") as file:
                model_data = json.load(file)
                
            # Add the model name to the dictionary (assuming the filename represents the model name)
            model_data["name"] = os.path.splitext(filename)[0].replace("_evaluation", "")
            
            # Append the model data to the list
            evaluations.append(model_data)
    return evaluations
