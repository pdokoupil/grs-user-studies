from collections import defaultdict
from bs4 import BeautifulSoup
import flask
import requests

from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.metrics.pairwise import pairwise_kernels

import numpy as np
import pandas as pd
import pickle

from .ml_data_loader import MLDataLoader, RatingUserFilter, RatingMovieFilter, MovieFilter, TagsFilter, TagsRatedMoviesFilter, RatingTagFilter
from .composed_func import ComposedFunc
from .rating_matrix_transform import SubtractMeanNormalize
from .popularity_sampling import PopularitySamplingElicitation, PopularitySamplingFromBucketsElicitation
# from ml_data_loader import MLDataLoader, RatingUserFilter, RatingMovieFilter, MovieFilter, TagsFilter, TagsRatedMoviesFilter, RatingTagFilter
# from composed_func import ComposedFunc
# from rating_matrix_transform import SubtractMeanNormalize
# from popularity_sampling import PopularitySamplingElicitation, PopularitySamplingFromBucketsElicitation
import flask_pluginkit

import time
import datetime as dt
from lenskit.algorithms import Recommender, als, item_knn, user_knn

MOST_RATED_MOVIES_THRESHOLD = 200
USERS_RATING_RATIO_THRESHOLD = 0.75
NUM_TAGS_PER_GROUP = 3
NUM_CLUSTERS = 6
NUM_CLUSTERS_TO_PICK = 1
NUM_MOVIES_PER_TAG = 2
MIN_NUM_TAG_OCCURRENCES = 50 # NUM_MOVIES_PER_TAG * NUM_TAGS_PER_GROUP # Calculation based because of the Deny list #10

MIN_RATINGS_PER_USER = 500
MIN_RATINGS_PER_MOVIE = 500
MIN_TAGS_PER_MOVIE = 50

import os
import functools
cluster_data = None

basedir = os.path.abspath(os.path.dirname(__file__))
print(f"File={__file__}")
print(f"Dirname: {os.path.dirname(__file__)}")

groups = None

# Loads the movielens dataset
@functools.cache
def load_ml_dataset(ml_variant="ml-latest"):
    ratings_path = os.path.join(basedir, "static", f"{ml_variant}/ratings.csv")
    movies_path = os.path.join(basedir, "static", f"{ml_variant}/movies.csv")
    rating_matrix_path = os.path.join(basedir, "static", f"{ml_variant}/rating_matrix.npy")
    tags_path = os.path.join(basedir, "static", f"{ml_variant}/tags.csv")
    links_path = os.path.join(basedir, "static", f"{ml_variant}/links.csv")
    
    start_time = time.perf_counter()
    loader = MLDataLoader(ratings_path, movies_path, tags_path, links_path,
        ComposedFunc([RatingMovieFilter(MIN_RATINGS_PER_MOVIE), RatingUserFilter(MIN_RATINGS_PER_USER), RatingTagFilter(MIN_TAGS_PER_MOVIE)]),
        MovieFilter(), TagsRatedMoviesFilter(), rating_matrix_path=rating_matrix_path
    )
    loader.load()
    print(f"## Loading took: {time.perf_counter() - start_time}")
    return loader

def load_data_1():
    global cluster_data
    
    if cluster_data is None:
        global groups    
        
        loader = load_ml_dataset()

        # Get dense
        start_time = time.perf_counter()
        dense_rm, most_rated_items_subset = gen_dense_rating_matrix(loader.rating_matrix)
        print(f"## Dense took: {time.perf_counter() - start_time}")
        start_time = time.perf_counter()
        most_rated_items_subset_ids = {loader.movie_index_to_id[i] for i in most_rated_items_subset}
        loader.apply_tag_filter(TagsFilter(most_rated_items_subset_ids, MIN_NUM_TAG_OCCURRENCES))
        print(f"## Tags filtering took: {time.perf_counter() - start_time}")
        
        # Normalize
        start_time = time.perf_counter()
        dense_rm = SubtractMeanNormalize()(dense_rm)
        print(f"Dense_rm = {dense_rm}")
        print(f"## Normalize took: {time.perf_counter() - start_time}")
        
        # Generate groups
        start_time = time.perf_counter()
        groups = gen_groups(dense_rm, NUM_CLUSTERS)
        print(f"Groups: {groups}")
        print(f"## Group generation took: {time.perf_counter() - start_time}")
        
        new_groups = dict()
        for idx, group in enumerate(groups):
            new_groups[most_rated_items_subset[idx]] = group
        groups = new_groups

        start_time = time.perf_counter()
        print(f"Formated groups: {groups}")
        group_labels = label_groups(groups, loader.tags, loader.tag_counts_per_movie)
        print(f"Group labels: {group_labels}")
        print(f"## Group labeling took: {time.perf_counter() - start_time}")

        movie_to_group = dict()
        for movie, group in groups.items():
            movie_to_group[movie] = group
        
        # Add most relevant movies
        start_time = time.perf_counter()
        cluster_data = []
        deny_list = set() # Across groups to prevent user confusion
        for group, group_tags in group_labels.items():
            print(f"Group={group} has tags={group_tags}\n\n")
            cluster_data.append(dict())
            cluster_data[-1]["tags"] = list()
            for tag in group_tags:
                most_rel = most_relevant_movies(group, movie_to_group, deny_list, tag, loader.tag_counts_per_movie, loader)
                movies_without_url = []
                d = {
                    "tag": tag,
                    "movies": []
                }
                for movie_idx in most_rel:
                    img_url = loader.get_image(movie_idx)
                    mov = {
                        "movie": loader.movie_index_to_description[movie_idx],
                        "url": img_url,
                        "movie_idx": movie_idx,
                        "description": loader.movie_index_to_description[movie_idx]
                    }
                    if not img_url:
                        movies_without_url.append(mov)
                    else:
                        d["movies"].append(mov)
                    if len(d["movies"]) >= NUM_MOVIES_PER_TAG:
                        print(f"Achieved: {len(d['movies'])} out of {NUM_MOVIES_PER_TAG} needed")
                        break
                
                if len(d["movies"]) < NUM_MOVIES_PER_TAG:
                    remaining = NUM_MOVIES_PER_TAG - len(d["movies"])
                    d["movies"].extend(movies_without_url[:remaining])
                
                # d = {
                #     "tag": tag,
                #     "movies": [
                #         {
                #             "movie": loader.movie_index_to_description[movie_idx],
                #             "url": get_image(loader.links_df.loc[loader.movie_index_to_id[movie_idx]].imdbId),
                #             "movie_idx": movie_idx,
                #             "description": loader.movie_index_to_description[movie_idx]
                #         }
                #         for movie_idx in most_rel
                #     ]
                # }
                #deny_list.update(most_rel)
                deny_list.update([m["movie_idx"] for m in d["movies"]])
                print(f"Deny list: {deny_list}")
                cluster_data[-1]["tags"].append(d)

        print(f"## Adding most relevant movies took: {time.perf_counter() - start_time}")
        # Result is a list of clusters, each cluster being a dict (JSON object) and having a list of movies and other properties
        print(f"Cluster data: {cluster_data}")
        return cluster_data
    return cluster_data

def load_data_2():
    
    cache_path = os.path.join(basedir, "static", "ml-latest", "data_cache.pckl")
    if os.path.exists(cache_path):
        print(f"Trying to load data cache from: {cache_path}")
        with open(cache_path, "rb") as f:
            loader = pickle.load(f)
    else:
        print("Cache not available, loading everything again")
        loader = load_ml_dataset()
        print(f"Caching the data to {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(loader, f)

    # Get list of items
    start_time = time.perf_counter()
    data = PopularitySamplingElicitation(loader.rating_matrix).get_initial_data()
    print(f"Getting initial data took: {time.perf_counter() - start_time}")
    
    #print([loader.movie_index_to_description[movie_idx] for movie_idx in data])
    #print([loader.movies_df.iloc[movie_idx].title for movie_idx in data])
    start_time = time.perf_counter()
    res = [loader.movie_index_to_description[movie_idx] for movie_idx in data]
    #print(f"Movies: {loader.movies_df.movieId.unique().shape}, {loader.links_df.movieId.unique().shape}")
    #imdbIds = [loader.links_df.loc[loader.movie_index_to_id[movie_idx]].imdbId for movie_idx in data]
    print(f"Up to now took: {time.perf_counter() - start_time}")
    start_time = time.perf_counter()
    #res_url = [loader.get_image(imdbId)["cover url"] for imdbId in imdbIds]
    res_url = [loader.get_image(movie_idx) for movie_idx in data]
    print(f"Getting urls took: {time.perf_counter() - start_time}")
    start_time = time.perf_counter()
    result = [{"movie": movie, "url": url, "movie_idx": str(movie_idx)} for movie, url, movie_idx in zip(res, res_url, data)]
    print(f"Result= {result} and took: {time.perf_counter() - start_time}")
    # Result is a list of movies, each movie being a dict (JSON object)
    return result

def load_data_3():
    loader = load_ml_dataset()
    # Get list of items
    data = PopularitySamplingFromBucketsElicitation(loader.rating_matrix, 5, [5]*5).get_initial_data()
    print([loader.movie_index_to_description[movie_idx] for movie_idx in data])
    print([loader.movies_df.iloc[movie_idx].title for movie_idx in data])
    res = [loader.movie_index_to_description[movie_idx] for movie_idx in data]
    #print(f"Movies: {loader.movies_df.movieId.unique().shape}, {loader.links_df.movieId.unique().shape}")
    #imdbIds = [loader.links_df.loc[loader.movie_index_to_id[movie_idx]].imdbId for movie_idx in data]
    #res_url = [loader.get_image(imdbId)["cover url"] for imdbId in imdbIds]
    res_url = [loader.get_image(movie_idx) for movie_idx in data]
    result = [{"movie": movie, "url": url, "movie_idx": str(movie_idx)} for movie, url, movie_idx in zip(res, res_url, data)]
    print(f"Result= {result}")
    # Result is a list of movies, each movie being a dict (JSON object)
    return result

def recommend_1(selected_cluster):
    loader = load_ml_dataset()

    start_time = time.perf_counter()
    dense_rm, most_rated_items_subset = gen_dense_rating_matrix(loader.rating_matrix)
    print(f"## Dense took: {time.perf_counter() - start_time}")
    start_time = time.perf_counter()
    most_rated_items_subset_ids = {loader.movie_index_to_id[i] for i in most_rated_items_subset}

    index_to_dense_index = {movie_index: dense_index for dense_index, movie_index in enumerate(most_rated_items_subset)}
    #loader.apply_tag_filter(TagsFilter(most_rated_items_subset_ids, MIN_NUM_TAG_OCCURRENCES))
    #print(f"## Tags filtering took: {time.perf_counter() - start_time}")
    
    # Normalize
    #start_time = time.perf_counter()
    #dense_rm = SubtractMeanNormalize()(dense_rm)
    #print(f"Dense_rm = {dense_rm}")
    #print(f"## Normalize took: {time.perf_counter() - start_time}")

    group_to_movies = dict()
    for movie, group in groups.items():
        if group not in group_to_movies:
            group_to_movies[group] = list()
        group_to_movies[group].append(movie)

    # For each user we have either 1 or 0 (to use them when averaging or not)
    users_to_select = np.zeros((dense_rm.shape[0], ), dtype=np.int32)
    movies_to_select = np.zeros((dense_rm.shape[1], ), dtype=np.bool)
    idx = 0
    for movie in group_to_movies[selected_cluster]:
        movie_dense_idx = index_to_dense_index[movie]
        users_to_select = np.logical_or(users_to_select, np.argmax(dense_rm, axis=1) == movie_dense_idx)
        movies_to_select[movie_dense_idx] = True
        idx += 1

    print(f"Users to select: {users_to_select}, SUM: {users_to_select.sum()}")
    print(f"Movies to select: {movies_to_select} shape: {movies_to_select.shape} and num positive: {movies_to_select.sum()}")
    #new_row = dense_rm[users_to_select, :].mean(axis=0)
    print(f"Shp1: {users_to_select.shape}, Shp2: {movies_to_select.shape}")
    new_row = dense_rm[np.ix_(users_to_select, movies_to_select)]
    print(f"Shp3: {new_row.shape}")
    print(f"New_row: {new_row.shape}, {new_row}")
    new_row = new_row.sum(axis=0) / (new_row > 0.0).sum(axis=0) # Mean over non-negative (rated) entries #new_row.mean(axis=0)
    print(f"New_row: {new_row.shape}, {new_row}")
    selected_movie_indices = np.where(movies_to_select)[0]

    #algo = user_knn.UserUser() #als.BiasedMF(50)
    algo = als.ImplicitMF(200, iterations=50)
    algo = Recommender.adapt(algo)
    max_user = loader.ratings_df.userId.max()
    ratings_df = loader.ratings_df

    

    print(ratings_df.tail(n=10))
    for i in range(new_row.shape[0]):
        selected_movie_idx = selected_movie_indices[i]
        movie_idx = most_rated_items_subset[selected_movie_idx]
        movie_id = loader.movie_index_to_id[movie_idx]
        if i % 100 == 0:
            print(f"I={i}, selected_movie_idx={selected_movie_idx} movie_idx={movie_idx}, movie_id={movie_id}")
        ratings_df.loc[ratings_df.index.max() + 1] = [max_user + 1, movie_id, new_row[i], dt.datetime.now()]

    # Implicit rating variant (comment for explicit)
    # TODO: move filtering after adding rows below? And filter >= 3.5?
    ratings_df = ratings_df[ratings_df.rating >= 3.5]
    # End Implicit

    # all_animation = loader.tags_df[loader.tags_df.tag == "animation"].movieId.unique()
    # selected_movies = [ 50872,  87876,   3745,   8360,  68358,    616,  96281, 177765,
    #     74553,   4306, 115617,  26662,   5218,  59784,   2355,  55442,
    #       594,   2137,   7228,  79091] #all_animation[:20]
    # for movie in selected_movies:
    #     ratings_df.loc[ratings_df.index.max() + 1] = [max_user + 1, movie, 5.0, dt.datetime.now()]

    print(ratings_df.tail(n=10))   
    ratings_df = ratings_df.rename(columns={"movieId": "item", "userId": "user"})

    # Ratings_df prepared, lets train
    print("Starting fit")
    algo = algo.fit(ratings_df)
    print("Starting prediction")
    #prediction = algo_als.predict_for_user(max_user + 1, ratings_df.item.unique())
    #print("Starting sort")
    #sorted_prediction = prediction.sort_values(ascending=False)[:10].index # Take top 10
    sorted_prediction = algo.recommend(max_user + 1, n=10)
    print(f"Sorted prediction: {sorted_prediction}")
    sorted_prediction = sorted_prediction.item
    
    
    top_k = [loader.movie_id_to_index[movie_id] for movie_id in sorted_prediction]
    # Result is a list of movies, each movie being a dict (JSON object)
    top_k_description = [loader.movie_index_to_description[movie_idx] for movie_idx in top_k]
    #imdbIds = [loader.links_df.loc[loader.movie_index_to_id[movie_idx]].imdbId for movie_idx in top_k]
    #top_k_url = [loader.get_image(imdbId)["cover url"] for imdbId in imdbIds]
    top_k_url = [loader.get_image(movie_idx) for movie_idx in top_k]

    res = [{"movie": movie, "url": url, "movie_idx": str(movie_idx)} for movie, url, movie_idx in zip(top_k_description, top_k_url, top_k)]
    print(res)
    return res

def recommend_2_3(selected_movies):
    loader = load_dataset()

    # algo_als = als.BiasedMF(10, iterations=5)
    algo = als.ImplicitMF(200, iterations=50)
    algo = Recommender.adapt(algo)

    max_user = loader.ratings_df.userId.max()
    ratings_df = loader.ratings_df
    for selected_movie in selected_movies:
        print(f"Selected movie: {selected_movie}")
        print(f"We have mapping to id: {selected_movie in loader.movie_index_to_id}")
        print(f"We have mapping to idx: {selected_movie in loader.movie_id_to_index}")
        print("Movie to index")
        print(loader.movie_id_to_index)
        print("Index to movie")
        print(loader.movie_index_to_id)
        ratings_df.loc[ratings_df.index.max() + 1] = [max_user + 1, loader.movie_index_to_id[selected_movie], 5.0, dt.datetime.now()]
    ratings_df = ratings_df.rename(columns={"movieId": "item", "userId": "user"})
    print(f"Df shape: {ratings_df.shape}")
    ratings_df = ratings_df[ratings_df.rating >= 4.0]
    print(f"Df shape after filter: {ratings_df.shape}")
    print(f"Extended ratings_df: {ratings_df[ratings_df.user == max_user + 1]}")
    print("Starting fit")
    algo = algo.fit(ratings_df)
    print("Starting prediction")
    #prediction = algo_als.predict_for_user(max_user + 1, ratings_df.item.unique())
    #print("Starting sort")
    #sorted_prediction = prediction.sort_values(ascending=False)[:10].index # Take top 10
    sorted_prediction = algo.recommend(max_user + 1, n=10)
    print(f"Sorted prediction: {sorted_prediction}")
    sorted_prediction = sorted_prediction.item
    
    top_k = [loader.movie_id_to_index[movie_id] for movie_id in sorted_prediction]
    # Result is a list of movies, each movie being a dict (JSON object)
    top_k_description = [loader.movie_index_to_description[movie_idx] for movie_idx in top_k]
    #imdbIds = [loader.links_df.loc[loader.movie_index_to_id[movie_idx]].imdbId for movie_idx in top_k]
    #top_k_url = [loader.get_image(imdbId)["cover url"] for imdbId in imdbIds]
    top_k_url = [loader.get_image(movie_idx) for movie_idx in top_k]

    return [{"movie": movie, "url": url, "movie_idx": str(movie_idx)} for movie, url, movie_idx in zip(top_k_description, top_k_url, top_k)]

# Takes rating matrix and returns dense copy
def gen_dense_rating_matrix(rating_matrix):
    #print(f"Rating matrix shape: {rating_matrix.shape}")
    #assert np.all(rating_matrix >= 0.0), "Rating matrix must be non-negative"
    # Number of times each item was rated
    ratings_per_item = np.sum(rating_matrix > 0.0, axis=0)
    most_rated_items = np.argsort(-ratings_per_item, kind="stable")
    # Take only MOST_RATED_MOVIES_THRESHOLD movies
    most_rated_items_subset = most_rated_items[:MOST_RATED_MOVIES_THRESHOLD]
    # Restrict rating matrix to the subset of items
    dense_rating_matrix = rating_matrix[:, most_rated_items_subset]
    # Restrict rating matrix to the subset of users
    n = np.minimum(MOST_RATED_MOVIES_THRESHOLD, most_rated_items_subset.size)
    per_user_rating_ratios = np.sum(dense_rating_matrix > 0, axis=1) / n
    selected_users = np.where(per_user_rating_ratios >= USERS_RATING_RATIO_THRESHOLD)[0]
    dense_rating_matrix = dense_rating_matrix[selected_users, :]
    assert dense_rating_matrix.ndim == rating_matrix.ndim, f"Dense rating matrix should preserve ndim: {dense_rating_matrix.shape}"
    assert np.all(dense_rating_matrix.shape <= rating_matrix.shape), f"Making dense rating matrix should not increase dimensions: {dense_rating_matrix.shape} vs. {rating_matrix.shape}"
    #print(f"Dense rating matrix shape: {dense_rating_matrix.shape}")
    return dense_rating_matrix, most_rated_items_subset


def gen_groups(rating_matrix, n_groups):
    similarities = cosine_similarity(rating_matrix.T)
    similarities = (similarities + 1) / 2
    clustering = SpectralClustering(n_groups, random_state=0, affinity="precomputed")
    groups = clustering.fit_predict(similarities)
    return groups

def tags_in_cluster(group_items, tags_per_item):
    tags = set()
    for item in group_items:
        tags.update(tags_per_item[item])
    return tags

# Relevance of tag to cluster/group of movies
def tag_relevance(tag, group_items, tag_counts_per_movie):
    rel = 0.0
    for item in group_items:
        rel += tag_counts_per_movie[item][tag]
    return rel

def most_relevant_movies(group, movie_to_group, deny_list, tag, tag_counts_per_movie, loader):
    movie_counts = dict()
    for movie, tag_counts in tag_counts_per_movie.items():
        if movie in deny_list or movie not in movie_to_group or movie_to_group[movie] != group:
            pass #movie_counts[movie] = -1
        else:
            movie_counts[movie] = tag_counts[tag]
    return sorted(movie_counts.keys(), key=lambda x: movie_counts[x], reverse=True)

def acc_per_cluster_tag_relevance(tag, group_to_items, tag_counts_per_movie):
    acc = 0.0
    for group, items in group_to_items.items():
        acc += tag_relevance(tag, items, tag_counts_per_movie)
    return acc

def acc_per_tag_tag_relevance(group_items, tag_counts_per_movie):
    acc = 0.0
    # for tag in group_tags:
    group_tags = set()
    for movie in group_items:
        for tag, tag_count in tag_counts_per_movie[movie].items():
            if tag_count > 0:
                group_tags.add(tag)
    #print(group_tags)
    for tag in group_tags:
        acc += tag_relevance(tag, group_items, tag_counts_per_movie)
    return acc

# Prepares description for each of the groups
def label_groups(group_assignment, tags, tag_counts_per_movie):
    group_to_items = dict()
    tags_per_group = dict()
    
    for item, group in group_assignment.items():
        if group not in group_to_items:
            tags_per_group[group] = set()
            group_to_items[group] = []
        group_to_items[group].append(item)

        for tag in tags:
            if tag in tag_counts_per_movie[item] and tag_counts_per_movie[item][tag] > 0:
                tags_per_group[group].add(tag)
    
    

    best_group_tags = dict()
    tag_deny_list = set()
    for group in set(group_assignment.values()):
        tag_prod = dict()
        for tag in tags_per_group[group]: #tags:
            if tag in tag_deny_list:
                pass
            else:
                d1 = acc_per_cluster_tag_relevance(tag, group_to_items, tag_counts_per_movie)
                if d1 == 0:
                    uniqueness = 0.0
                else:
                    uniqueness = tag_relevance(tag, group_to_items[group], tag_counts_per_movie) / d1
                d2 = acc_per_tag_tag_relevance(group_to_items[group], tag_counts_per_movie)
                if d2 == 0:
                    relevance = 0.0
                else:
                    relevance = tag_relevance(tag, group_to_items[group], tag_counts_per_movie) / d2
                tag_prod[tag] = uniqueness * relevance
        #print(tag_prod)
        best_tags = sorted(tag_prod.keys(), key=lambda x: tag_prod[x], reverse=True)
        #print(f"Best tags for group={group} are: {best_tags}")
        best_group_tags[group] = best_tags[:NUM_TAGS_PER_GROUP]
    return best_group_tags


# Shape should be [num_users, num_items]
def elicit_preferences(rating_matrix):
    assert type(rating_matrix) is np.ndarray, f"Expecting numpy ndarray rating matrix: {rating_matrix}"
    assert rating_matrix.ndim == 2, f"Expecting 2D rating matrix: {rating_matrix.shape}"

    
    #dense_rating_matrix = gen_dense_rating_matrix(rating_matrix)
    #print(dense_rating_matrix)

if __name__ == "__main__":
    #y = load_data_1()
    #x = recommend_1(1)
    load_data_2()