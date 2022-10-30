import pandas as pd
import numpy as np
from collections import defaultdict
import os

# Movie-lens data loader

class RatingUserFilter:
    def __init__(self, min_ratings_per_user):
        self.min_ratings_per_user = min_ratings_per_user

    def __call__(self, ratings_df):
        # First filter out users who gave <= 1 ratings
        ratings_df = ratings_df[ratings_df['userId'].map(ratings_df['userId'].value_counts()) >= self.min_ratings_per_user]
        ratings_df = ratings_df.reset_index(drop=True)
        print(f"Ratings shape after user filtering: {ratings_df.shape}, n_users = {ratings_df.userId.unique().size}, n_items = {ratings_df.movieId.unique().size}")
        return ratings_df
        

class RatingMovieFilter:
    def __init__(self, min_ratings_per_movie):
        self.min_ratings_per_movie = min_ratings_per_movie
    def __call__(self, ratings_df):
        # Filter out users that were rated <= 1 times
        ratings_df = ratings_df[ratings_df['movieId'].map(ratings_df['movieId'].value_counts()) >= self.min_ratings_per_movie]
        ratings_df = ratings_df.reset_index(drop=True)
        print(f"Ratings shape after item filtering: {ratings_df.shape}, n_users = {ratings_df.userId.unique().size}, n_items = {ratings_df.movieId.unique().size}")
        return ratings_df

class MovieFilter:
    def __call__(self, movies_df, ratings_df):
        # We are only interested in movies for which we hav
        movies_df = movies_df[movies_df.movieId.isin(ratings_df.movieId.unique())]
        movies_df = movies_df.reset_index(drop=True)
        return movies_df

class TagsFilter:
    def __init__(self, most_rated_items_subset_ids, min_num_tag_occurrences):
        self.min_num_tag_occurrences = MIN_NUM_TAG_OCCURRENCES
    def __call__(self, tags_df):
        tags_df = tags_df[tags_df.movieId.isin(self.most_rated_items_subset_ids)]
        print(f"Tags_df shape: {tags_df.shape}")
        tags_df = tags_df[tags_df['tag'].map(tags_df['tag'].value_counts()) >= self.min_num_tag_occurrences]
        print(f"Tags_df shape: {tags_df.shape}")
        return tags_df

class MLDataLoader:
    def __init__(self, ratings_path, movies_path, tags_path,
        ratings_df_filter = None,  movies_df_filter = None,
        tags_df_filter = None, rating_matrix_path = None):

        self.ratings_path = ratings_path
        self.movies_path = movies_path
        self.tags_path = tags_path
        self.ratings_df_filter = ratings_df_filter
        self.movies_df_filter = movies_df_filter
        self.tags_df_filter = tags_df_filter
        self.ratings_matrix_path = rating_matrix_path

        self.ratings_df = None
        self.movies_df = None
        self.tags_df = None
        self.rating_matrix = None
        self.movie_index_to_id = None
        self.movie_id_to_index = None
        self.num_movies = None
        self.num_users = None
        self.user_to_user_index = None
        
    def load(self):
        if self.ratings_df:
            print("Already loaded")
            return
        
        self.ratings_df = pd.read_csv(self.ratings_path)
        print(f"Ratings shape: {self.ratings_df.shape}, n_users = {self.ratings_df.userId.unique().size}, n_items = {self.ratings_df.movieId.unique().size}")
        # Filter rating dataframe
        if self.ratings_df_filter:
            self.ratings_df = self.ratings_df_filter(self.ratings_df)
        
        
        self.movies_df = pd.read_csv(self.movies_path)
        # Filter movies dataframe
        if self.movies_df_filter:
            self.movies_df = self.movies_df_filter(self.movies_df, self.ratings_df)

        self.tags_df = pd.read_csv(self.tags_path)
        if self.tags_df_filter:
            self.tags_df = self.tags_df_filter(self.tags_df, most_rated_items_subset_ids)

        self.movie_index_to_id = pd.Series(self.movies_df.movieId.values,index=self.movies_df.index).to_dict()
        self.movie_id_to_index = pd.Series(self.movies_df.index,index=self.movies_df.movieId.values).to_dict()
        num_movies = len(self.movie_id_to_index)


        unique_users = self.ratings_df.userId.unique()
        num_users = unique_users.size
        
        self.user_to_user_index = dict(zip(unique_users, range(num_users)))

        # Attempt to load cached rating matrix
        if self.rating_matrix_path and os.path.exists(self.rating_matrix_path):
            rating_matrix = np.load(self.rating_matrix_path)
        else:
            rating_matrix = np.zeros(shape=(num_users, num_movies), dtype=np.float32)
            for row_idx, row in self.ratings_df.iterrows():
                # 25 prints per data frame
                if row_idx % (self.ratings_df.shape[0] // 25) == 0:
                    print(row_idx)
                rating_matrix[self.user_to_user_index[row.userId], self.movie_id_to_index[row.movieId]] = row.rating
            np.save(self.rating_matrix_path, rating_matrix)

        

        
        
        # Maps movie index to text description
        self.movies_df["description"] = self.movies_df.title + ' ' + self.movies_df.genres
        self.movie_index_to_description = dict(zip(self.movies_df.index, self.movies_df.description))
        
        # Set of all unique tags
        self.tags = set(self.tags_df.tag.unique())
        
        # Maps movie index to tag counts per movie
        tag_counts_per_movie = { movie_index : defaultdict(int) for movie_index in self.movie_index_to_id.keys() }
        for group_name, group_df in self.tags_df.groupby("movieId"):
            for _, row in group_df.iterrows():
                tag_counts_per_movie[self.movie_id_to_index[group_name]][row.tag] += 1
