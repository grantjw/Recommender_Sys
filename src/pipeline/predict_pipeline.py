import sys
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import load_object  # Ensure this function is defined in utils.py to load the pickled objects
import pandas as pd
from src.components.data_ingestion import DataIngestionConfig

# Load movies and ratings dataframes
class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join('artifacts', 'model.pkl')
        self.mapping_files_path = os.path.join('artifacts', 'mappings.pkl')
        self.id_to_genre_mapping_path = os.path.join('artifacts', 'id_to_genre_mapping.pkl')
        self.title_id_mapping_path = os.path.join('artifacts', 'title_id_mapping.pkl')
        self.id_to_title_mapping_path = os.path.join('artifacts', 'id_to_title_mapping.pkl')  

        self.Q = load_object(self.model_path)
        mappings = load_object(self.mapping_files_path) 
        self.movie_mapper = mappings['movie_mapper']
        self.movie_inv_mapper = mappings['movie_inv_mapper']
        self.user_mapper = mappings['user_mapper']
        self.title_id_mapping = load_object(self.title_id_mapping_path)
        self.id_to_title_mapping = load_object(self.id_to_title_mapping_path)  # Load the mapping
        self.id_to_genre_mapping = load_object(self.id_to_genre_mapping_path)  # Load the mapping
        # Load Movie Lens Data
        self.movies = pd.read_csv(DataIngestionConfig().movies_data_path)
        self.ratings = pd.read_csv(DataIngestionConfig().ratings_data_path)

    def get_movie_id_by_title(self, title):
        return self.title_id_mapping.get(title)

    def get_title_by_movie_id(self, movie_id):
        """
        Retrieves a movie title for a given ID.
        """
        return self.id_to_title_mapping.get(movie_id)

    def get_genre_by_movie_id(self, movie_id):
        """
        Retrieves the genre(s) for a given movie ID.

        Parameters:
            movie_id (int): The ID of the movie.
        
        Returns:
            str: The genre(s) of the movie, or "Unknown" if not found.
        """
        return self.id_to_genre_mapping.get(movie_id, "Unknown")

    def get_all_movie_titles(self):
        """
        Returns a list of all movie titles.
        """
        return sorted(self.title_id_mapping.keys())  # Assuming title_id_mapping is {title: id}

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
    
    def get_user_recommendations(self, user_id, k=10):
        try:
            user_index = self.user_mapper.get(user_id)  # Convert user_id to user_index
            if user_index is None:
                print(f"User with ID {user_id} not found")
                return []

            # Load user-based predicted ratings
            predicted_ratings = load_object(os.path.join('artifacts', 'predicted_ratings.pkl'))
            
            # It's important to ensure that the indexing here matches the length of user_predicted_ratings
            # and that we're operating on a DataFrame copy to avoid SettingWithCopyWarning.
            movies_included = self.movies.loc[self.movies['movieId'].isin(self.movie_mapper.keys())].copy()
            
            # Assigning predicted ratings to the movies DataFrame. Ensure lengths match to avoid errors.
            user_predicted_ratings = predicted_ratings[user_index, :len(movies_included)]
            movies_included.loc[:, 'predicted_rating'] = user_predicted_ratings
            
            # Filtering out movies already rated by the user to avoid recommending them again
            rated_movies = self.ratings[self.ratings['userId'] == user_id]['movieId'].unique()
            recommended_movies = movies_included[~movies_included['movieId'].isin(rated_movies)]
            
            # Sort by predicted rating and take the top k
            top_recommendations = recommended_movies.sort_values(by='predicted_rating', ascending=False).head(k)
            
            # Converting to a list of dicts for compatibility with Jinja template rendering
            return top_recommendations[['movieId', 'title', 'genres', 'predicted_rating']].to_dict(orient='records')

        except KeyError as e:
            print(f"Error: {e}")
            return []
