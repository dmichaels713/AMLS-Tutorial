# Pull in Libaries
import json
import pickle
import argparse
import os
from fbprophet import Prophet
import numpy as np
import pandas as pd
import azureml.train.automl
import joblib
from azureml.core.model import Model
from azureml.core import Workspace, Datastore, Dataset, Run

# Pass in Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model-name', dest="model_name", required=True)
parser.add_argument('--scoring-directory', dest="scoring_directory", required=True)
parser.add_argument('--input-data', dest="input_data", required=True)
args = parser.parse_args() 

global model
# Retrieve Model
model_path = Model.get_model_path(model_name = args.model_name)
model = joblib.load(model_path)

def main():
    #create output directories if they do not exist
    os.makedirs(args.scoring_directory, exist_ok=True)

    # Pull in an input dataset
    DataPath = args.input_data

    # Convert to Pandas Dataframe
    DataDF = pd.read_csv(DataPath)

    # Score Data
    scoredDataResults = pd.Series(model.predict(DataDF))

    # Join Results to original data
    scoredData = DataDF
    scoredData['Label'] = scoredDataResults

    #Save Results
    scoredFileName = "prediction"
    scoredPath = os.path.join(args.scoring_directory, scoredFileName)
    scoredData.to_csv(scoredPath, index = False)
    
if __name__ == '__main__':
    main()
