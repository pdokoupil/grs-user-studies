import os
import pandas as pd
import numpy as np

MIN_RATINGS_PER_USER = 500
MIN_RATINGS_PER_MOVIE = 500

# Popularity-sampling based implementation of preference elicitation
class PopularitySamplingElicitation:
    
    def __init__(self, basedir):
        self.basedir = basedir

        # Load Rating matrix
        ratings_path = os.path.join(basedir, "static", "ml-latest/ratings.csv")
        ratings_df = pd.read_csv(ratings_path)
        
        # Filter rating matrix
        # First filter out users who gave <= 1 ratings
        ratings_df = ratings_df[ratings_df['userId'].map(ratings_df['userId'].value_counts()) >= MIN_RATINGS_PER_USER]
        print(f"Ratings shape after user filtering: {ratings_df.shape}, n_users = {ratings_df.userId.unique().size}, n_items = {ratings_df.movieId.unique().size}")
        # Then filter out users that were rated <= 1 times
        ratings_df = ratings_df[ratings_df['movieId'].map(ratings_df['movieId'].value_counts()) >= MIN_RATINGS_PER_MOVIE]
        ratings_df = ratings_df.reset_index(drop=True)
        print(f"Ratings shape after item filtering: {ratings_df.shape}, n_users = {ratings_df.userId.unique().size}, n_items = {ratings_df.movieId.unique().size}")


        movies_path = os.path.join(basedir, "static", "ml-latest/movies.csv")
        movies_df = pd.read_csv(movies_path)
        movies_df = movies_df[movies_df.movieId.isin(ratings_df.movieId.unique())]
        movies_df = movies_df.reset_index(drop=True)

        movie_index_to_id = pd.Series(movies_df.movieId.values,index=movies_df.index).to_dict()
        movie_id_to_index = pd.Series(movies_df.index,index=movies_df.movieId.values).to_dict()
        num_movies = len(movie_id_to_index)


        unique_users = ratings_df.userId.unique()
        num_users = unique_users.size
        
        user_to_user_index = dict(zip(unique_users, range(num_users)))

        rating_matrix_path = os.path.join(basedir, "static", "ml-latest/rating_matrix.npy")
        if os.path.exists(rating_matrix_path):
            rating_matrix = np.load(rating_matrix_path)
        else:
            rating_matrix = np.zeros(shape=(num_users, num_movies), dtype=np.float32)
            for row_idx, row in ratings_df.iterrows():
                if row_idx % 100000 == 0:
                    print(row_idx)
                rating_matrix[user_to_user_index[row.userId], movie_id_to_index[row.movieId]] = row.rating
            np.save(rating_matrix_path, rating_matrix)

        
        popularities = self._calculate_item_popularities(rating_matrix)
        p_popularities = popularities / popularities.sum()
        print(f"Popularities = {popularities}")
        print(f"p_popularities = {p_popularities}")
        s = np.random.choice(np.arange(rating_matrix.shape[1]), p=p_popularities, size=10)
        print(f"Sample: {s}")
        print(f"RM shape: {rating_matrix.shape}")
        print(f"Sample items: {popularities[s]}")

    def _calculate_item_popularities(self, rating_matrix):
        return np.sum(rating_matrix > 0.0, axis=0) / rating_matrix.shape[0]


    # Returns data to be shown to the user
    def get_initial_data(self):
        pass

    # Based on user selection of initial data, return vector with ratings of all the items
    def get_preferences(self, user_selection):
        pass


if __name__ == "__main__":
    x = PopularitySamplingElicitation(os.path.abspath(os.path.dirname(__file__)))
    x.get_initial_data()