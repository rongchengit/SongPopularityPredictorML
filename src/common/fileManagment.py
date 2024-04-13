import json
import logging
import os
import joblib
import numpy as np

# Create a logger
logger = logging.getLogger("fileManagement")

TRAINED_MODELS_DIRECTORY = 'trainedModels'
EVALUATION_RESULTS_DIRECTORY = 'evaluationResults'

# Get all Trained Model versions and returns them in descending order
def getVersions():
    versions = [d for d in os.listdir(TRAINED_MODELS_DIRECTORY) if d.startswith('v')]
    
    # Sort the versions in descending order
    sorted_versions = sorted(versions, key=lambda x: int(x[1:]), reverse=True)
    
    return sorted_versions

# Gets the latest version plus one
def getNewVersion():
    # Get the list of existing versions
    versions = getVersions()

    if versions:
        # Extract the numeric version from the latest version directory
        latest_version = int(versions[0][1:])
        return f'v{latest_version + 1}'
    else:
        return 'v1'

# Get the newest version directory, unless a version is provided
def getVersionDirectory(base_directory, version=None):
    # Ensure the base directory exists
    os.makedirs(base_directory, exist_ok=True)

    if version is None:
        version = getVersions()[-1]

    version_directory = os.path.join(base_directory, version)

    # Create the version directory if it doesn't exist
    os.makedirs(version_directory, exist_ok=True)

    return version_directory, version

def saveModel(model, name, dataset_length, version=None):
    version_directory, version = getVersionDirectory(TRAINED_MODELS_DIRECTORY, version)
    
    model_filename = name + '.joblib'
    full_path = os.path.join(version_directory, model_filename)
    joblib.dump(model, full_path)
    
    metadata_filename = 'metadata.json'
    metadata_path = os.path.join(version_directory, metadata_filename)
    
    if not os.path.exists(metadata_path):
        metadata = {'datasetLen': dataset_length}
        with open(metadata_path, 'w') as file:
            json.dump(metadata, file, indent=4)
        logger.debug(f'Metadata saved to {metadata_path}')
    
    logger.debug(f'Model saved to {full_path}')
    
    return version

# Load the model via name (by default gets the newest model version)
def loadModel(name, version=None):    
    if version is None:
        version = getVersions()[-1]
    
    version_directory = os.path.join(TRAINED_MODELS_DIRECTORY, version)
    model_filename = name + '.joblib'
    full_path = os.path.join(version_directory, model_filename)
    
    if os.path.exists(full_path):
        model = joblib.load(full_path)
        logger.info(f'Model loaded from {full_path}')
        return model, version
    else:
        logger.error(f'Model file {full_path} not found.')
        return None, None

def saveEvaluation(evaluationMetrics, model_type, suffix, version=None):   
    version_directory, version = getVersionDirectory(EVALUATION_RESULTS_DIRECTORY, version)
    
    results_filename = f"{model_type}_{suffix}.json"
    full_path = os.path.join(version_directory, results_filename)
    
    with open(full_path, 'w') as file:
        json.dump(evaluationMetrics, file, indent=4)
    
    logger.debug(f'Evaluation results saved to {full_path}')

def loadEvaluations(suffix, version):
    evaluations = []
    directory = os.path.join(EVALUATION_RESULTS_DIRECTORY, version)
    # Iterate over the files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(suffix + ".json"):
            file_path = os.path.join(directory, filename)
            
            # Load the JSON data from the file
            with open(file_path, "r") as file:
                model_data = json.load(file)
                
            # Add the model name to the dictionary (assuming the filename represents the model name)
            model_data["name"] = os.path.splitext(filename)[0].replace("_"+ suffix, "")
            
            # Append the model data to the list
            evaluations.append(model_data)
    return evaluations

def saveNumpyFile(data, model_type, suffix, version=None):
    version_directory, version = getVersionDirectory(EVALUATION_RESULTS_DIRECTORY, version)
    
    results_filename = f"{model_type}_{suffix}.npy"
    full_path = os.path.join(version_directory, results_filename)
    
    np.save(full_path, data)
    
    logger.debug(f'Results saved to {full_path}')

def loadNumpyFile(suffix, version, model_type=None):
    directory = os.path.join(EVALUATION_RESULTS_DIRECTORY, version)
    # Iterate over the files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(suffix + ".npy") and (not model_type or filename.startswith(model_type)):
            file_path = os.path.join(directory, filename)
            return np.load(file_path, allow_pickle=True)

def loadMetadata(version):
    directory = os.path.join(TRAINED_MODELS_DIRECTORY, version)
    # Iterate over the files in the directory
    for filename in os.listdir(directory):
        if filename.endswith("metadata.json"):
            file_path = os.path.join(directory, filename)
            
            # Load the JSON data from the file
            with open(file_path, "r") as file:
                return json.load(file)