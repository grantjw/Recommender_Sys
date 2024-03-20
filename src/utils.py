import os
import sys

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException

def save_object(obj, file_path):
    """
    Saves the object to the file path.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(svd):
    """
    Evaluates the model.
    """
    try:
        explained_variance = svd.explained_variance_ratio_
        #get the variance explained by the first 20 components
        cumulative_variance = explained_variance[:20].sum()
        return cumulative_variance
        
    except Exception as e:
        raise CustomException(e, sys)