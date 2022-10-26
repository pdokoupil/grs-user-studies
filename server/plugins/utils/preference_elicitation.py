from collections import defaultdict
import flask

from sklearn.cluster import SpectralClustering

import numpy as np
import pandas as pd

import flask_pluginkit

MOST_RATED_MOVIES_THRESHOLD = 200
USERS_RATING_RATIO_THRESHOLD = 0.75
NUM_TAGS_PER_GROUP = 3
NUM_CLUSTERS = 6
NUM_CLUSTERS_TO_PICK = 1
NUM_MOVIES_PER_TAG = 2

import os
cluster_data = None

basedir = os.path.abspath(os.path.dirname(__file__))
print(f"File={__file__}")
print(f"Dirname: {os.path.dirname(__file__)}")

def load_data():
    global cluster_data
    if cluster_data is None:
        # Load info about movies
        # movies_url = flask.url_for(
        #     "assets",
        #     plugin_name="utils",
        #     filename="ml-latest-small/movies.csv",
        #     _external=False,
        # )
        movies_path = os.path.join(basedir, "static", "ml-latest-small/movies.csv")
        movies_df = pd.read_csv(movies_path)
        movie_index_to_id = pd.Series(movies_df.movieId.values,index=movies_df.index).to_dict()
        movie_id_to_index = pd.Series(movies_df.index,index=movies_df.movieId.values).to_dict()
        num_movies = len(movie_id_to_index)

        # Load Rating matrix
        ratings_path = os.path.join(basedir, "static", "ml-latest-small/ratings.csv")
        ratings_df = pd.read_csv(ratings_path)
        unique_users = ratings_df.userId.unique()
        num_users = unique_users.size
        
        user_to_user_index = dict(zip(unique_users, range(num_users)))

        rating_matrix = np.zeros(shape=(num_users, num_movies), dtype=np.float32)
        for _, row in ratings_df.iterrows():
            rating_matrix[user_to_user_index[row.userId], movie_id_to_index[row.movieId]] = row.rating

        tags_path = os.path.join(basedir, "static", "ml-latest-small/tags.csv")
        tags_df = pd.read_csv(tags_path)
        
        
        # Maps movie index to text description
        movies_df["description"] = movies_df.title + ' ' + movies_df.genres
        movie_index_to_description = dict(zip(movies_df.index, movies_df.description))
        
        # Set of all unique tags
        tags = set(tags_df.tag.unique())
        
        # Maps movie index to tag counts per movie
        tag_counts_per_movie = { movie_index : defaultdict(int) for movie_index in movie_index_to_id.keys() }
        for group_name, group_df in tags_df.groupby("movieId"):
            for _, row in group_df.iterrows():
                tag_counts_per_movie[movie_id_to_index[group_name]][row.tag] += 1


        # Get dense
        dense_rm = gen_dense_rating_matrix(rating_matrix)
        # Normalize
        dense_rm = subtract_mean_normalize(dense_rm)
        # Generate groups
        groups = gen_groups(dense_rm, NUM_CLUSTERS)
        group_labels = label_groups(groups, tags, tag_counts_per_movie)
        # Add most relevant movies
        cluster_data = {}
        for group, group_tags in group_labels.items():
            cluster_data[group] = dict()
            for tag in group_tags:
                cluster_data[group][tag] = [movie_index_to_description[movie_idx] for movie_idx in most_relevant_movies(tag, tag_counts_per_movie)]
        return cluster_data

# Takes rating matrix and returns dense copy
def gen_dense_rating_matrix(rating_matrix):
    print(f"Rating matrix shape: {rating_matrix.shape}")
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
    print(f"Dense rating matrix shape: {dense_rating_matrix.shape}")
    return dense_rating_matrix

# Normalize the rating matrix by subtracting mean rating of each user
def subtract_mean_normalize(rating_matrix):
    return rating_matrix - rating_matrix.mean(axis=1, keepdims=True)

def gen_groups(rating_matrix, n_groups):
    clustering = SpectralClustering(n_groups, random_state=0)
    groups = clustering.fit_predict(rating_matrix.T)
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

def most_relevant_movies(tag, tag_counts_per_movie):
    movie_counts = dict()
    for movie, tag_counts in tag_counts_per_movie.items():
        movie_counts[movie] = tag_counts[tag]
    return sorted(movie_counts.keys(), key=lambda x: movie_counts[x], reverse=True)[:NUM_MOVIES_PER_TAG]

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
    print(group_tags)
    for tag in group_tags:
        acc += tag_relevance(tag, group_items, tag_counts_per_movie)
    return acc

# Prepares description for each of the groups
def label_groups(group_assignment, tags, tag_counts_per_movie):
    group_to_items = dict()
    for item, group in enumerate(group_assignment):
        if group not in group_to_items:
            group_to_items[group] = []
        group_to_items[group].append(item)
    best_group_tags = dict()
    for group in set(group_assignment):
        tag_prod = dict()
        for tag in tags:
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
        print(tag_prod)
        best_tags = sorted(tag_prod.keys(), key=lambda x: tag_prod[x], reverse=True)
        print(f"Best tags for group={group} are: {best_tags}")
        best_group_tags[group] = best_tags[:NUM_TAGS_PER_GROUP]
    return best_group_tags


# Shape should be [num_users, num_items]
def elicit_preferences(rating_matrix):
    assert type(rating_matrix) is np.ndarray, f"Expecting numpy ndarray rating matrix: {rating_matrix}"
    assert rating_matrix.ndim == 2, f"Expecting 2D rating matrix: {rating_matrix.shape}"

    dense_rating_matrix = gen_dense_rating_matrix(rating_matrix)
    print(dense_rating_matrix)


