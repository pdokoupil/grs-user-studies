from collections import defaultdict
import flask

from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.metrics.pairwise import pairwise_kernels

import numpy as np
import pandas as pd

from ml_data_loader import MLDataLoader, RatingUserFilter, RatingMovieFilter, MovieFilter, TagsFilter
from composed_func import ComposedFunc
from rating_matrix_transform import SubtractMeanNormalize

import flask_pluginkit

MOST_RATED_MOVIES_THRESHOLD = 200
USERS_RATING_RATIO_THRESHOLD = 0.75
NUM_TAGS_PER_GROUP = 3
NUM_CLUSTERS = 6
NUM_CLUSTERS_TO_PICK = 1
NUM_MOVIES_PER_TAG = 2
MIN_NUM_TAG_OCCURRENCES = 10

MIN_RATINGS_PER_USER = 500
MIN_RATINGS_PER_MOVIE = 500

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
        
        # Load Rating matrix
        ratings_path = os.path.join(basedir, "static", "ml-latest/ratings.csv")
        movies_path = os.path.join(basedir, "static", "ml-latest/movies.csv")
        rating_matrix_path = os.path.join(basedir, "static", "ml-latest/rating_matrix.npy")
        tags_path = os.path.join(basedir, "static", "ml-latest/tags.csv")
        
    
        loader = MLDataLoader(ratings_path, movies_path, tags_path,
            ComposedFunc([RatingUserFilter(MIN_RATINGS_PER_USER), RatingMovieFilter(MIN_RATINGS_PER_MOVIE)]),
            MovieFilter(), None, rating_matrix_path=rating_matrix_path
        )
        loader.load()
        



        # Get dense
        dense_rm, most_rated_items_subset = gen_dense_rating_matrix(loader.rating_matrix)
        most_rated_items_subset_ids = {movie_index_to_id[i] for i in most_rated_items_subset}
        loader.tags_df = TagsFilter(most_rated_items_subset_ids, MIN_NUM_TAG_OCCURRENCES)(loader.tags_df)
        
        # Normalize
        dense_rm = SubtractMeanNormalize()(dense_rm)
        print(f"Dense_rm = {dense_rm}")
        # Generate groups
        groups = gen_groups(dense_rm, NUM_CLUSTERS)
        print(f"Groups: {groups}")
        
        new_groups = dict()
        for idx, group in enumerate(groups):
            new_groups[most_rated_items_subset[idx]] = group
        groups = new_groups

        print(f"Formated groups: {groups}")
        group_labels = label_groups(groups, tags, tag_counts_per_movie)
        print(f"Group labels: {group_labels}")

        movie_to_group = dict()
        for movie, group in groups.items():
            movie_to_group[movie] = group
        
        # Add most relevant movies
        cluster_data = []
        deny_list = set() # Across groups to prevent user confusion
        for group, group_tags in group_labels.items():
            print(f"Group={group} has tags={group_tags}\n\n")
            cluster_data.append(dict())
            cluster_data[-1]["tags"] = list()
            for tag in group_tags:
                most_rel = most_relevant_movies(group, movie_to_group, deny_list, tag, tag_counts_per_movie)
                d = {
                    "tag": tag,
                    "movies": [movie_index_to_description[movie_idx] for movie_idx in most_rel]
                }
                deny_list.update(most_rel)
                print(f"Deny list: {deny_list}")
                cluster_data[-1]["tags"].append(d)
        return cluster_data
    return cluster_data

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

def most_relevant_movies(group, movie_to_group, deny_list, tag, tag_counts_per_movie):
    movie_counts = dict()
    for movie, tag_counts in tag_counts_per_movie.items():
        if movie in deny_list or movie not in movie_to_group or movie_to_group[movie] != group:
            pass #movie_counts[movie] = -1
        else:
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

    dense_rating_matrix = gen_dense_rating_matrix(rating_matrix)
    #print(dense_rating_matrix)


if __name__ == "__main__":
    print("ABCD")
    x = load_data()