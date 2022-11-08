import glob
import time
import imdb
import pandas as pd
import numpy as np
from collections import defaultdict
import os

import pickle

# Movie-lens data loader

class RatingUserFilter:
    def __init__(self, min_ratings_per_user):
        self.min_ratings_per_user = min_ratings_per_user

    def __call__(self, ratings_df, *args, **kwargs):
        # First filter out users who gave <= 1 ratings
        ratings_df = ratings_df[ratings_df['userId'].map(ratings_df['userId'].value_counts()) >= self.min_ratings_per_user]
        ratings_df = ratings_df.reset_index(drop=True)
        print(f"Ratings shape after user filtering: {ratings_df.shape}, n_users = {ratings_df.userId.unique().size}, n_items = {ratings_df.movieId.unique().size}")
        return ratings_df
        

class RatingMovieFilter:
    def __init__(self, min_ratings_per_movie):
        self.min_ratings_per_movie = min_ratings_per_movie
    def __call__(self, ratings_df, *args, **kwargs):
        # Filter out users that were rated <= 1 times
        ratings_df = ratings_df[ratings_df['movieId'].map(ratings_df['movieId'].value_counts()) >= self.min_ratings_per_movie]
        ratings_df = ratings_df.reset_index(drop=True)
        print(f"Ratings shape after item filtering: {ratings_df.shape}, n_users = {ratings_df.userId.unique().size}, n_items = {ratings_df.movieId.unique().size}")
        return ratings_df

class RatingTagFilter:
    def __init__(self, min_tags_per_movie):
        self.min_tags_per_movie = min_tags_per_movie
    def __call__(self, ratings_df, loader, *args, **kwargs):
        # Filter out movies that do not have enough tags (we do not want those movies to end up in the dense pool for group based elicitation)
        # Even if they have many ratings
        tags_per_movie = loader.tags_df.groupby("movieId")["movieId"].count()
        tags_per_movie = tags_per_movie[tags_per_movie > self.min_tags_per_movie]
        print(f"Ratings shape before tag filtering: {ratings_df.shape}, n_users = {ratings_df.userId.unique().size}, n_items = {ratings_df.movieId.unique().size}")
        ratings_df = ratings_df[ratings_df.movieId.isin(tags_per_movie)]
        ratings_df = ratings_df.reset_index(drop=True)
        print(f"Ratings shape after tag filtering: {ratings_df.shape}, n_users = {ratings_df.userId.unique().size}, n_items = {ratings_df.movieId.unique().size}")
        return ratings_df

class MovieFilter:
    def __call__(self, movies_df, ratings_df, *args, **kwargs):
        # We are only interested in movies for which we hav
        movies_df = movies_df[movies_df.movieId.isin(ratings_df.movieId.unique())]
        movies_df = movies_df.reset_index(drop=True)
        return movies_df

# Just filters out tags that are not present on any of the rated movies
class TagsRatedMoviesFilter:
    def __call__(self, tags_df, ratings_df, *args, **kwargs):
        print(f"TagsRatedMoviesFilter before: {tags_df.shape}")
        tags_df = tags_df[tags_df.movieId.isin(ratings_df.movieId.unique())]
        tags_df = tags_df.reset_index(drop=True)
        print(f"TagsRatedMoviesFilter after: {tags_df.shape}")
        return tags_df

class TagsFilter:
    def __init__(self, most_rated_items_subset_ids, min_num_tag_occurrences):
        self.min_num_tag_occurrences = min_num_tag_occurrences
        self.most_rated_items_subset_ids = most_rated_items_subset_ids
    def __call__(self, tags_df, *args, **kwargs):
        # For the purpose of group-based preference elicitation we are only interested in tags that occurr in dense subset of items
        # over which the groups are defined
        tags_df = tags_df[tags_df.movieId.isin(self.most_rated_items_subset_ids)]
        print(f"Tags_df shape: {tags_df.shape}")
        # We also only consider tags that have enough occurrences, otherwise we will not be able to find enough representants (movies) for each tag
        tags_df = tags_df[tags_df['tag'].map(tags_df['tag'].value_counts()) >= self.min_num_tag_occurrences]
        print(f"Tags_df shape: {tags_df.shape}")
        tags_df = tags_df.reset_index(drop=True)
        return tags_df

class MLDataLoader:
    def __init__(self, ratings_path, movies_path, tags_path, links_path,
        ratings_df_filter = None,  movies_df_filter = None,
        tags_df_filter = None, rating_matrix_path = None):

        self.ratings_path = ratings_path
        self.movies_path = movies_path
        self.tags_path = tags_path
        self.ratings_df_filter = ratings_df_filter
        self.movies_df_filter = movies_df_filter
        self.tags_df_filter = tags_df_filter
        self.links_path = links_path
        self.rating_matrix_path = rating_matrix_path

        self.ratings_df = None
        self.movies_df = None
        self.tags_df = None
        self.links_df = None
        self.rating_matrix = None
        self.movie_index_to_id = None
        self.movie_id_to_index = None
        self.num_movies = None
        self.num_users = None
        self.user_to_user_index = None
        self.movie_index_to_description = None
        self.tag_counts_per_movie = None

        self.access = imdb.IMDb()
        self.movie_index_to_url = dict()

    def _get_image(self, imdbId):
        print("@@ Get movie function")
        try:
            print("@@ Try")
            #return self.access.get_movie(imdbId)["cover url"]
            return self.access.get_movie(imdbId)["full-size cover url"]
        except Exception as e:
            print("@@ Exception")
            return ""

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle imdb
        del state["access"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.access = imdb.IMDb()

    def get_image(self, movie_idx):
        return self.movie_index_to_url[movie_idx]
        
    def apply_tag_filter(self, tag_filter, *args, **kwargs):
        self.tags_df = tag_filter(self.tags_df, *args, **kwargs)

        # Set of all unique tags
        self.tags = set(self.tags_df.tag.unique())
        print(f"Num tags={len(self.tags)}")
        # Maps movie index to tag counts per movie
        self.tag_counts_per_movie = { movie_index : defaultdict(int) for movie_index in self.movie_index_to_id.keys() }
        for group_name, group_df in self.tags_df.groupby("movieId"):
            for _, row in group_df.iterrows():
                self.tag_counts_per_movie[self.movie_id_to_index[group_name]][row.tag] += 1

    def load(self):

        
        # Prepare cache paths
        # rating_cache_dir = os.path.dirname(self.ratings_path)
        # ratings_cache = os.path.join(self.ratings_path, os.path.splitext()[0], "_cache.pckl")
        # rating_matrix_cache = os.path.join(rating_cache_dir, "rm_cache.pckl")
        # movies_cache = os.path.join(self.ratings_path, os.path.splitext()[0], "_cache.pckl")
        # tags_cache = os.path.join(self.ratings_path, os.path.splitext()[0], "_cache.pckl")
        # links_cache = os.path.join(self.ratings_path, os.path.splitext()[0], "_cache.pckl")

        # if use_cache:
        #     print(f"use_cache is True, trying to check cache first")

        #     cache_paths = [
        #         ratings_cache,
        #         rating_matrix_cache,
        #         movies_cache,
        #         tags_cache,
        #         links_cache
        #     ]


        #     # To prevent out-of-sync of the cache, we follow the assumption that either all the cache files
        #     # are present, or all of them will be recomputed (i.e. just one missing cache file means that everything)
        #     # will be recomputed
        #     for cache_path in cache_paths:
        #         result = os.path.exists(cache_path)
        #         print(f"Checking existence of cache file: {cache_path}, result: {result}")
            
        #         if not result:
        #             print(f"Some cache files were missing, recompute everything")
        #             break

        #     self.ratings_df = 
            


        #     return
        
        #### Data Loading ####

        # Load ratings
        self.ratings_df = pd.read_csv(self.ratings_path)
        print(f"Ratings shape: {self.ratings_df.shape}, n_users = {self.ratings_df.userId.unique().size}, n_items = {self.ratings_df.movieId.unique().size}")
        
        # Load tags and convert them to lower case
        self.tags_df = pd.read_csv(self.tags_path)
        self.tags_df.tag = self.tags_df.tag.str.casefold()

        # Load movies
        self.movies_df = pd.read_csv(self.movies_path)

        # Load links
        self.links_df = pd.read_csv(self.links_path, index_col=0)
        
        #### Filtering ####

        # Filter rating dataframe
        if self.ratings_df_filter:
            self.ratings_df = self.ratings_df_filter(self.ratings_df, self)

        # Filter movies dataframe
        if self.movies_df_filter:
            self.movies_df = self.movies_df_filter(self.movies_df, self.ratings_df)

        self.movie_index_to_id = pd.Series(self.movies_df.movieId.values,index=self.movies_df.index).to_dict()
        self.movie_id_to_index = pd.Series(self.movies_df.index,index=self.movies_df.movieId.values).to_dict()
        num_movies = len(self.movie_id_to_index)

        # Filter tags dataframe
        if self.tags_df_filter:
            self.apply_tag_filter(self.tags_df_filter, self.ratings_df)


        unique_users = self.ratings_df.userId.unique()
        num_users = unique_users.size
        
        self.user_to_user_index = dict(zip(unique_users, range(num_users)))

        # Attempt to load cached rating matrix
        if self.rating_matrix_path and os.path.exists(self.rating_matrix_path):
            self.rating_matrix = np.load(self.rating_matrix_path)
        else:
            self.rating_matrix = np.zeros(shape=(num_users, num_movies), dtype=np.float32)
            for row_idx, row in self.ratings_df.iterrows():
                # 25 prints per data frame
                if row_idx % (self.ratings_df.shape[0] // 25) == 0:
                    print(row_idx)
                self.rating_matrix[self.user_to_user_index[row.userId], self.movie_id_to_index[row.movieId]] = row.rating
            np.save(self.rating_matrix_path, self.rating_matrix)

        

        # Maps movie index to text description
        self.movies_df["description"] = self.movies_df.title + ' ' + self.movies_df.genres
        self.movie_index_to_description = dict(zip(self.movies_df.index, self.movies_df.description))
        
        # Prepare urls
        for movie_id, row in self.links_df.iterrows():
            if movie_id not in self.movie_id_to_index:
                continue
            movie_idx = self.movie_id_to_index[movie_id]
            self.movie_index_to_url[movie_idx] = self._get_image(row.imdbId)

        return True