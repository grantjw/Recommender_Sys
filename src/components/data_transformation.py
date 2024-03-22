import os
import sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from scipy.sparse import save_npz

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    sparse_matrix_path: str = os.path.join('artifacts', 'sparse_matrix.npz')
    mapping_files_path: str = os.path.join('artifacts', 'mappings.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def create_sparse_matrix(self, ratings):
        """
        Generates a sparse matrix from ratings dataframe.
        """
        M = ratings['userId'].nunique()
        N = ratings['movieId'].nunique()

        user_mapper = dict(zip(np.unique(ratings["userId"]), list(range(M))))
        movie_mapper = dict(zip(np.unique(ratings["movieId"]), list(range(N))))
        
        user_inv_mapper = dict(zip(list(range(M)), np.unique(ratings["userId"])))
        movie_inv_mapper = dict(zip(list(range(N)), np.unique(ratings["movieId"])))
        
        user_index = [user_mapper[i] for i in ratings['userId']]
        item_index = [movie_mapper[i] for i in ratings['movieId']]

        X = csr_matrix((ratings["rating"], (user_index, item_index)), shape=(M, N))
        
        return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

    def initiate_data_transformation(self, ratings_data_path):
        """
        Initiates the data transformation process.
        """
        try:
            ratings = pd.read_csv(ratings_data_path)
            logging.info("Read the ratings dataset as dataframe")

            X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = self.create_sparse_matrix(ratings)
            
            # Correctly saving the sparse matrix
            os.makedirs(os.path.dirname(self.data_transformation_config.sparse_matrix_path), exist_ok=True)
            save_npz(self.data_transformation_config.sparse_matrix_path, X)
            
            mappings = {
                'user_mapper': user_mapper,
                'movie_mapper': movie_mapper,
                'user_inv_mapper': user_inv_mapper,
                'movie_inv_mapper': movie_inv_mapper
            }
            save_object(file_path=self.data_transformation_config.mapping_files_path, obj=mappings)
            
            logging.info("Sparse matrix and mappings have been saved successfully")
            
            return (
                    X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper,
                    self.data_transformation_config.sparse_matrix_path, 
                    self.data_transformation_config.mapping_files_path)
        except Exception as e:
            raise CustomException(e, sys)
    def save_title_id_mapping(self, movies_data_path):
        try:
            movies = pd.read_csv(movies_data_path)
            """
            Saves a mapping from movie titles to movie IDs.
            """
            title_id_mapping = dict(zip(movies['title'], movies['movieId']))
            save_object(title_id_mapping, os.path.join('artifacts', 'title_id_mapping.pkl'))
            logging.info("Title to ID mapping has been saved successfully")
        except Exception as e:
            raise CustomException(e, sys)

    # Add to the data_transformation.py script

    def save_id_to_title_mapping(self, movies_data_path):
        try:
            movies = pd.read_csv(movies_data_path)
            """
            Saves a mapping from movie IDs to titles.
            """
            id_to_title_mapping = dict(zip(movies['movieId'], movies['title']))
            save_object(id_to_title_mapping, os.path.join('artifacts', 'id_to_title_mapping.pkl'))
            logging.info("ID to title mapping has been saved successfully")
        except Exception as e:
            raise CustomException(e, sys)
    
    def save_id_to_genre_mapping(self, movies_data_path):#
        """
        Saves a mapping from movie IDs to genres.
        """
        try:
            movies = pd.read_csv(movies_data_path)
            id_to_genre_mapping = dict(zip(movies['movieId'], movies['genres']))
            save_object(id_to_genre_mapping, os.path.join('artifacts', 'id_to_genre_mapping.pkl'))
            logging.info("ID to genre mapping has been saved successfully")
        except Exception as e:
            raise CustomException(e, sys)
