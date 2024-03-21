import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    ratings_data_path: str=os.path.join('artifacts',"ratings.csv")
    movies_data_path: str=os.path.join('artifacts',"movies.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            ratings=pd.read_csv('notebook/data/ratings.csv')
            movies=pd.read_csv('notebook/data/movies.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.ratings_data_path),exist_ok=True)

            ratings.to_csv(self.ingestion_config.ratings_data_path,index=False,header=True)
            movies.to_csv(self.ingestion_config.movies_data_path,index=False,header=True)

            logging.info("Inmgestion of the data is completed")

            return(
                self.ingestion_config.ratings_data_path,
                self.ingestion_config.movies_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    ratings_data_path,movies_data_path=obj.initiate_data_ingestion()
    
    data_transformation=DataTransformation()
    X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper, _, _= data_transformation.initiate_data_transformation(ratings_data_path)
    data_transformation.save_title_id_mapping(movies_data_path)
    data_transformation.save_id_to_title_mapping(movies_data_path)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper))
    modeltrainer.train_user_based_svd(X)
