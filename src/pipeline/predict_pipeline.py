import sys
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import load_object  # Ensure this function is defined in utils.py to load the pickled objects

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join('artifacts', 'model.pkl')
        self.mapping_files_path = os.path.join('artifacts', 'mappings.pkl')
        self.title_id_mapping_path = os.path.join('artifacts', 'title_id_mapping.pkl')
        self.id_to_title_mapping_path = os.path.join('artifacts', 'id_to_title_mapping.pkl')  # Add this line
        
        self.Q = load_object(self.model_path)
        mappings = load_object(self.mapping_files_path) 
        self.movie_mapper = mappings['movie_mapper']
        self.movie_inv_mapper = mappings['movie_inv_mapper']
        self.title_id_mapping = load_object(self.title_id_mapping_path)
        self.id_to_title_mapping = load_object(self.id_to_title_mapping_path)  # Load the mapping

    def get_movie_id_by_title(self, title):
        return self.title_id_mapping.get(title)

    def get_title_by_movie_id(self, movie_id):
        """
        Retrieves a movie title for a given ID.
        """
        return self.id_to_title_mapping.get(movie_id)
    def find_similar_movies(self, movie_id, k=10):
        """
        Finds similar movies based on the cosine similarity of the SVD-transformed matrix.
        
        Parameters:
            movie_id (int): The ID of the movie for which to find similar movies.
            k (int): The number of similar movies to return.
        
        Returns:
            list of int: The IDs of the k most similar movies.
        """
        if movie_id not in self.movie_mapper:
            return []
        
        movie_idx = self.movie_mapper[movie_id]
        sim_scores = cosine_similarity(self.Q[movie_idx:movie_idx + 1], self.Q).flatten()
        indices = np.argsort(sim_scores)[-k-1:-1][::-1]  # Exclude the movie itself and reverse to get top scores
        
        # Convert matrix indices back to movie IDs, excluding the query movie itself
        similar_movie_ids = [self.movie_inv_mapper[x] for x in indices if x != movie_idx]
        
        return similar_movie_ids
