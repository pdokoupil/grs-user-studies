import datetime
import glob
import time
import imdb
import pandas as pd
import numpy as np
from collections import defaultdict
import os

import pickle

from scipy.spatial.distance import squareform, pdist

from PIL import Image
from io import BytesIO
import requests

from app import pm

# Movie-lens data loader

class RatingUserFilter:
    def __init__(self, min_ratings_per_user):
        self.min_ratings_per_user = min_ratings_per_user

    def __call__(self, loader):
        # First filter out users who gave <= 1 ratings
        loader.ratings_df = loader.ratings_df[loader.ratings_df['userId'].map(loader.ratings_df['userId'].value_counts()) >= self.min_ratings_per_user]
        loader.ratings_df = loader.ratings_df.reset_index(drop=True)
        print(f"Ratings shape after user filtering: {loader.ratings_df.shape}, n_users = {loader.ratings_df.userId.unique().size}, n_items = {loader.ratings_df.movieId.unique().size}")
        
# Filters out all low ratings
class RatingLowFilter:
    def __init__(self, min_rating):
        self.min_rating = min_rating
    def __call__(self, loader):
        loader.ratings_df = loader.ratings_df[loader.ratings_df.rating >= self.min_rating]
        loader.ratings_df = loader.ratings_df.reset_index(drop=True)

class RatingMovieFilter:
    def __init__(self, min_ratings_per_movie):
        self.min_ratings_per_movie = min_ratings_per_movie
    def __call__(self, loader):
        # Filter out users that were rated <= 1 times
        loader.ratings_df = loader.ratings_df[loader.ratings_df['movieId'].map(loader.ratings_df['movieId'].value_counts()) >= self.min_ratings_per_movie]
        loader.ratings_df = loader.ratings_df.reset_index(drop=True)
        print(f"Ratings shape after item filtering: {loader.ratings_df.shape}, n_users = {loader.ratings_df.userId.unique().size}, n_items = {loader.ratings_df.movieId.unique().size}")

class RatingTagFilter:
    def __init__(self, min_tags_per_movie):
        self.min_tags_per_movie = min_tags_per_movie
    def __call__(self, loader):
        # Filter out movies that do not have enough tags (we do not want those movies to end up in the dense pool for group based elicitation)
        # Even if they have many ratings
        tags_per_movie = loader.tags_df.groupby("movieId")["movieId"].count()
        tags_per_movie = tags_per_movie[tags_per_movie > self.min_tags_per_movie]
        print(f"Ratings shape before tag filtering: {loader.ratings_df.shape}, n_users = {loader.ratings_df.userId.unique().size}, n_items = {loader.ratings_df.movieId.unique().size}")
        loader.ratings_df = loader.ratings_df[loader.ratings_df.movieId.isin(tags_per_movie)]
        loader.ratings_df = loader.ratings_df.reset_index(drop=True)
        print(f"Ratings shape after tag filtering: {loader.ratings_df.shape}, n_users = {loader.ratings_df.userId.unique().size}, n_items = {loader.ratings_df.movieId.unique().size}")

class RatedMovieFilter:
    def __call__(self, loader):
        # We are only interested in movies for which we hav
        loader.movies_df = loader.movies_df[loader.movies_df.movieId.isin(loader.ratings_df.movieId.unique())]
        loader.movies_df = loader.movies_df.reset_index(drop=True)

# Filters out all ratings of movies that do not have enough ratings per year
class RatingsPerYearFilter:
    def __init__(self, min_ratings_per_year):
        self.min_ratings_per_year = min_ratings_per_year

    def __call__(self, loader):
        movies_df_indexed = loader.movies_df.set_index("movieId")

        # Add column with age of each movie
        movies_df_indexed.loc[:, "age"] = movies_df_indexed.year.max() - movies_df_indexed.year
        
        # Calculate number of ratings per year for each of the movies
        loader.ratings_df.loc[:, "ratings_per_year"] = loader.ratings_df['movieId'].map(loader.ratings_df['movieId'].value_counts()) / loader.ratings_df['movieId'].map(movies_df_indexed["age"])
        
        # Filter out movies that do not have enough yearly ratings
        loader.ratings_df = loader.ratings_df[loader.ratings_df.ratings_per_year >= self.min_ratings_per_year]

class MovieFilterByYear:
    def __init__(self, min_year):
        self.min_year = min_year
        
    def _parse_year(self, x):
        x = x.split("(")
        if len(x) <= 1:
            return 0
        try:
            return int(x[-1].split(")")[0])
        except:
            return 0

    def __call__(self, loader):
        # Filter out unrated movies and old movies
        # Add year column      
        loader.movies_df.loc[:, "year"] = loader.movies_df.title.apply(self._parse_year)
        loader.movies_df = loader.movies_df[loader.movies_df.year >= self.min_year]
        loader.movies_df = loader.movies_df.reset_index(drop=True)

class RatingFilterOld:
    def __init__(self, oldest_rating_year):
        self.oldest_rating_year = oldest_rating_year
    def __call__(self, loader):
        # Marker for oldest rating
        oldest_rating = datetime.datetime(year=self.oldest_rating_year, month=1, day=1, tzinfo=datetime.timezone.utc).timestamp()
        # Filter ratings that are too old
        loader.ratings_df = loader.ratings_df[loader.ratings_df.timestamp > oldest_rating]
        #loader.ratings_df = loader.ratings_df.reset_index(drop=True)

# Just filters out tags that are not present on any of the rated movies
class TagsRatedMoviesFilter:
    def __call__(self, loader):
        print(f"TagsRatedMoviesFilter before: {loader.tags_df.shape}")
        loader.tags_df = loader.tags_df[loader.tags_df.movieId.isin(loader.ratings_df.movieId.unique())]
        loader.tags_df = loader.tags_df.reset_index(drop=True)
        print(f"TagsRatedMoviesFilter after: {loader.tags_df.shape}")

class TagsFilter:
    def __init__(self, most_rated_items_subset_ids, min_num_tag_occurrences):
        self.min_num_tag_occurrences = min_num_tag_occurrences
        self.most_rated_items_subset_ids = most_rated_items_subset_ids
    def __call__(self, loader):
        # For the purpose of group-based preference elicitation we are only interested in tags that occurr in dense subset of items
        # over which the groups are defined
        loader.tags_df = loader.tags_df[loader.tags_df.movieId.isin(self.most_rated_items_subset_ids)]
        print(f"Tags_df shape: {loader.tags_df.shape}")
        # We also only consider tags that have enough occurrences, otherwise we will not be able to find enough representants (movies) for each tag
        loader.tags_df = loader.tags_df[loader.tags_df['tag'].map(loader.tags_df['tag'].value_counts()) >= self.min_num_tag_occurrences]
        print(f"Tags_df shape: {loader.tags_df.shape}")
        loader.tags_df = loader.tags_df.reset_index(drop=True)

class LinkFilter:
    def __call__(self, loader):
        loader.links_df = loader.links_df[loader.links_df.index.isin((loader.movies_df.movieId))]

class MLDataLoader:
    def __init__(self, ratings_path, movies_path, tags_path, links_path,
        filters = None, rating_matrix_path = None, img_dir_path = None):

        self.ratings_path = ratings_path
        self.movies_path = movies_path
        self.tags_path = tags_path
        self.filters = filters
        self.links_path = links_path
        self.rating_matrix_path = rating_matrix_path

        self.ratings_df = None
        self.movies_df = None
        self.movies_df_indexed = None
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
        self.similarity_matrix = None

        self.img_dir_path = img_dir_path        

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

    # Download all the images
    def download_images(self):
        i = 0
        start_time = time.perf_counter()
        for movie_idx, url in self.movie_index_to_url.items():
            
            if i % 100 == 0:
                print(f"{i}/{len(self.movie_index_to_url)} images were downloaded, took: {time.perf_counter() - start_time}")
                start_time = time.perf_counter()

            movie_id = self.movie_index_to_id[movie_idx]
            
            err = False
            try:
                resp = requests.get(url, stream=True)
            except:
                err = True

            if err or resp.status_code != 200:
                print(f"Error downloading image from url={url} for movie_id={movie_id}")
                # Try once more retrieving fresh url for the given movie
                imdbId = self.links_df.loc[movie_id].imdbId
                new_url = self._get_image(imdbId)

                err = False
                try:
                    resp = requests.get(new_url, stream=True)
                except:
                    err = True

                if err or resp.status_code != 200:
                    print(f"Even new URL={new_url} failed")
                    continue # Ignore the movie
                else:
                    print("New URL succeeded")

            img = Image.open(BytesIO(resp.content))
            width, height = img.size
            TARGET_WIDTH = 200
            coef = TARGET_WIDTH / width
            new_height = int(height * coef)
            img = img.resize((TARGET_WIDTH, new_height),Image.ANTIALIAS)
            img.save(os.path.join(self.img_dir_path, f'{movie_id}.jpg'), quality=90)

            i += 1

    def get_image(self, movie_idx):
        if self.img_dir_path:
            # Use local version of images
            return pm.emit_assets('utils', f'ml-latest/img/{self.movie_index_to_id[movie_idx]}.jpg')

        return self.movie_index_to_url[movie_idx]
        #movie_id = self.movie_index_to_id[movie_idx]
        #return self._get_image(self.links_df.loc[movie_id].imdbId)

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

    # Passing local_movie_images as parameter to prevent cache changes
    # If local_movie_images==True, we will use local files instead of image urls inside img uris
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

        # # Filter rating dataframe
        # if self.ratings_df_filter:
        #     self.ratings_df = self.ratings_df_filter(self.ratings_df, self)

        # # Filter movies dataframe
        # if self.movies_df_filter:
        #     self.movies_df = self.movies_df_filter(self.movies_df, self.ratings_df)

        for f in self.filters:
            f(self)

        self.movie_index_to_id = pd.Series(self.movies_df.movieId.values,index=self.movies_df.index).to_dict()
        self.movie_id_to_index = pd.Series(self.movies_df.index,index=self.movies_df.movieId.values).to_dict()
        num_movies = len(self.movie_id_to_index)

        # # Filter tags dataframe
        # if self.tags_df_filter:
        #     self.apply_tag_filter(self.tags_df_filter, self.ratings_df)


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

        self.similarity_matrix = np.float32(squareform(pdist(self.rating_matrix.T, "cosine")))

        # Maps movie index to text description
        self.movies_df["description"] = self.movies_df.title + ' ' + self.movies_df.genres
        self.movie_index_to_description = dict(zip(self.movies_df.index, self.movies_df.description))
        

        self.movies_df_indexed = self.movies_df.set_index("movieId")


        # Prepare urls
        i = 0
        print(f"Links shape: {self.links_df.shape}")
        for movie_id, row in self.links_df.iterrows():
            if i % 100 == 0:
                print(f"i={i} out of: {self.links_df.shape}")
            i += 1
            if movie_id not in self.movie_id_to_index:
                continue
            movie_idx = self.movie_id_to_index[movie_id]
            self.movie_index_to_url[movie_idx] = self._get_image(row.imdbId)


        if self.img_dir_path:
            self.download_images()        

        return True