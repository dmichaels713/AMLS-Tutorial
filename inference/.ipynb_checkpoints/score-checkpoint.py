# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import pickle
import numpy as np
import pandas as pd
import azureml.train.automl
from sklearn.externals import joblib
from azureml.core.model import Model

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


input_sample = pd.DataFrame(data=[{'RowID': 0, 'GR': 99.0, 'smoothedGR': 104.0, 'FirstRow': 'First', 'smoothedGR1': None, 'smoothedGR-1': 104.0, 'smoothedGR2': None, 'smoothedGR-2': 104.1, 'smoothedGR3': None, 'smoothedGR-3': 104.1, 'smoothedGR4': None, 'smoothedGR-4': 104.2, 'smoothedDifference': 0.0, 'smoothedDifference_SMA': -2.8, 'smoothedDifference_SMA_4': None, 'smoothedDifference_SMA_8': None, 'smoothedDifference_SMA_12': None, 'smoothedDifference_SMA_16': None, 'smoothedDifference_SMA_20': None, 'smoothedDifference_SMA_24': None, 'smoothedDifference_SMA_28': None, 'smoothedDifference_SMA_32': None, 'smoothedDifference_SMA_36': None, 'smoothedDifference_SMA_40': None, 'smoothedDifference_SMA_FORWARD_4': -1.1, 'smoothedDifference_SMA_FORWARD_8': 4.5, 'smoothedDifference_SMA_FORWARD_12': -3.2, 'smoothedDifference_SMA_FORWARD_16': 0.7, 'smoothedDifference_SMA_FORWARD_20': 0.3, 'smoothedDifference_SMA_FORWARD_24': 1.6, 'smoothedDifference_SMA_FORWARD_28': -0.8, 'smoothedDifference_SMA_FORWARD_32': -2.0, 'smoothedDifference_SMA_FORWARD_36': 2.6, 'smoothedDifference_SMA_FORWARD_40': -1.6}])
output_sample = np.array([0])


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = Model.get_model_path(model_name = 'AutoML37a31f86512')
    model = joblib.load(model_path)


@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
