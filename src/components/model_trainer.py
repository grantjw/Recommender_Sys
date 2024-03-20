import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import os 

from src.exception import CustomException
from src.logger import logging


from src.utils import save_object,evaluate_model
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self, X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper):
        try:
            logging.info("train svd")
            svd = TruncatedSVD(n_components=20, n_iter=7)
            Q= svd.fit_transform(X.T)
            
            score=evaluate_model(svd=svd)
            logging.info("Model evaluation is completed")

            # if score does not exist, raise an exception
            if score is None:
                raise CustomException("Cumulative explained variance does not exist")            
        

            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=Q)
            logging.info("Model is saved")
            return score 
        
        except Exception as e:
            raise CustomException(e,sys)
        
