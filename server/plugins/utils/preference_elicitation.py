# import numba
# numba.config.THREADING_LAYER = "tbb"

import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.metrics.pairwise import pairwise_kernels

import numpy as np
import pickle


import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
tf.random.set_seed(42)
import tensorflow_recommenders as tfrs

# from .ml_data_loader import MLDataLoader, RatingUserFilter, RatingMovieFilter, RatedMovieFilter, TagsFilter, TagsRatedMoviesFilter, RatingTagFilter, MovieFilterByYear, RatingFilterOld, RatingsPerYearFilter, RatingLowFilter, LinkFilter
# from .composed_func import ComposedFunc
# from .rating_matrix_transform import SubtractMeanNormalize
# from .popularity_sampling import PopularitySamplingElicitation, PopularitySamplingFromBucketsElicitation
# from .tfrs_model import get_model_25m
from ml_data_loader import MLDataLoader, RatingUserFilter, RatingMovieFilter, RatedMovieFilter, TagsFilter, TagsRatedMoviesFilter, RatingTagFilter, MovieFilterByYear, RatingFilterOld, RatingsPerYearFilter, RatingLowFilter, LinkFilter
from composed_func import ComposedFunc
from rating_matrix_transform import SubtractMeanNormalize
from popularity_sampling import PopularitySamplingElicitation, PopularitySamplingFromBucketsElicitation
from multi_obj_sampling import MultiObjectiveSamplingFromBucketsElicitation
from tfrs_model import get_model_25m

import requests

import time
import datetime as dt
from lenskit.algorithms import Recommender, als, item_knn, user_knn
from sklearn.preprocessing import QuantileTransformer
from PIL import Image
from io import BytesIO


from rlprop_wrapper import RLPropWrapper
from normalization.identity import identity
from normalization.cdf import cdf
from normalization.standardization import standardization
from mandate_allocation.exactly_proportional_fuzzy_dhondt_2 import exactly_proportional_fuzzy_dhondt_2
from mandate_allocation.weighted_average_strategy import weighted_average_strategy

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


USE_LOCAL_IMAGES = True

import os
import functools
cluster_data = None

basedir = os.path.abspath(os.path.dirname(__file__))
print(f"File={__file__}")
print(f"Dirname: {os.path.dirname(__file__)}")

groups = None

result_layout_variants = [
    "columns",
    "column-single",
    "rows",
    "row-single",
    "row-single-scrollable"
]

# Loads the movielens dataset
@functools.lru_cache(maxsize=None)
def load_ml_dataset(ml_variant="ml-latest"):
    cache_path = os.path.join(basedir, "static", "ml-latest", "data_cache.pckl")
    if os.path.exists(cache_path):
        print(f"Trying to load data cache from: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    else:
        print("Cache not available, loading everything again")
        
        ratings_path = os.path.join(basedir, "static", f"{ml_variant}/ratings.csv")
        movies_path = os.path.join(basedir, "static", f"{ml_variant}/movies.csv")
        rating_matrix_path = os.path.join(basedir, "static", f"{ml_variant}/rating_matrix.npy")
        tags_path = os.path.join(basedir, "static", f"{ml_variant}/tags.csv")
        links_path = os.path.join(basedir, "static", f"{ml_variant}/links.csv")
        img_dir_path = os.path.join(basedir, "static", "ml-latest", "img")
        
        start_time = time.perf_counter()
        # loader = MLDataLoader(ratings_path, movies_path, tags_path, links_path,
        #     ComposedFunc([RatingMovieFilter(MIN_RATINGS_PER_MOVIE), RatingUserFilter(MIN_RATINGS_PER_USER), RatingTagFilter(MIN_TAGS_PER_MOVIE)]),
        #     RatedMovieFilter(), TagsRatedMoviesFilter(), rating_matrix_path=rating_matrix_path
        # )
        loader = MLDataLoader(ratings_path, movies_path, tags_path, links_path,
            [RatingLowFilter(4.0), MovieFilterByYear(1990), RatingFilterOld(2010), RatingsPerYearFilter(50.0), RatingUserFilter(100), RatedMovieFilter(), LinkFilter()],
            rating_matrix_path=rating_matrix_path, img_dir_path=img_dir_path
        )
        loader.load()
        print(f"## Loading took: {time.perf_counter() - start_time}")

        print(f"Caching the data to {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(loader, f)

        return loader

# Decorator that will cache the result of the decorated function after first call
# difference from functools.cache is that it ignores the value of the parameters
def compute_once(func):
    result = None
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal result
        if not result:
            result = func(*args, **kwargs)
        return result
    return wrapper

@compute_once
def prepare_tf_data(loader):
    ratings_df = loader.ratings_df.copy()

    # Add movie_title
    ratings_df.loc[:, "movie_title"] = ratings_df.movieId.map(loader.movies_df_indexed.title)

    # Rename column and cast to string
    ratings_df = ratings_df.rename(columns={"userId": "user_id"})
    ratings_df.user_id = ratings_df.user_id.astype(str)

    ratings = tf.data.Dataset.from_tensor_slices(dict(ratings_df[["user_id", "movie_title"]]))
    movies = tf.data.Dataset.from_tensor_slices(dict(loader.movies_df.rename(columns={"title": "movie_title"})[["movie_title"]])).map(lambda x: x["movie_title"])

    import numpy as np
    tf.random.set_seed(42)
    shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

    train_size = int(ratings_df.shape[0] * 0.85)
    test_size = int(ratings_df.shape[0] - train_size)
    print(f"Train_size={train_size}, test_size={test_size}")


    #train = shuffled.take(train_size)
    #test = shuffled.skip(train_size).take(test_size)
    # Take everything as train
    train = shuffled

    movie_titles = movies.batch(1_000)
    user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))

    new_user = str(max([int(x) for x in unique_user_ids]) + 1)
    unique_user_ids = np.concatenate([unique_user_ids, np.array([new_user])])

    cached_train = train.shuffle(100_000).batch(8192).cache()

    return unique_user_ids, unique_movie_titles, movies, cached_train, train

def prepare_tf_model(loader):

    unique_user_ids, unique_movie_titles, movies, cached_train, train = prepare_tf_data(loader)
    model = get_model_25m(unique_user_ids, unique_movie_titles, movies)
    cache_path = os.path.join(basedir, "static", "ml-latest", "tf_weights_cache")

    # Try load
    try:
        model.load_weights(cache_path)
    except tf.errors.NotFoundError as ex:
        model.fit(cached_train, epochs=5)
        model.save_weights(cache_path)

    return model, train

# def load_data_1():
#     global cluster_data
    
#     if cluster_data is None:
#         global groups    
        
#         loader = load_ml_dataset()

#         # Get dense
#         start_time = time.perf_counter()
#         dense_rm, most_rated_items_subset = gen_dense_rating_matrix(loader.rating_matrix)
#         print(f"## Dense took: {time.perf_counter() - start_time}")
#         start_time = time.perf_counter()
#         most_rated_items_subset_ids = {loader.movie_index_to_id[i] for i in most_rated_items_subset}
#         loader.apply_tag_filter(TagsFilter(most_rated_items_subset_ids, MIN_NUM_TAG_OCCURRENCES))
#         print(f"## Tags filtering took: {time.perf_counter() - start_time}")
        
#         # Normalize
#         start_time = time.perf_counter()
#         dense_rm = SubtractMeanNormalize()(dense_rm)
#         print(f"Dense_rm = {dense_rm}")
#         print(f"## Normalize took: {time.perf_counter() - start_time}")
        
#         # Generate groups
#         start_time = time.perf_counter()
#         groups = gen_groups(dense_rm, NUM_CLUSTERS)
#         print(f"Groups: {groups}")
#         print(f"## Group generation took: {time.perf_counter() - start_time}")
        
#         new_groups = dict()
#         for idx, group in enumerate(groups):
#             new_groups[most_rated_items_subset[idx]] = group
#         groups = new_groups

#         start_time = time.perf_counter()
#         print(f"Formated groups: {groups}")
#         group_labels = label_groups(groups, loader.tags, loader.tag_counts_per_movie)
#         print(f"Group labels: {group_labels}")
#         print(f"## Group labeling took: {time.perf_counter() - start_time}")

#         movie_to_group = dict()
#         for movie, group in groups.items():
#             movie_to_group[movie] = group
        
#         # Add most relevant movies
#         start_time = time.perf_counter()
#         cluster_data = []
#         deny_list = set() # Across groups to prevent user confusion
#         for group, group_tags in group_labels.items():
#             print(f"Group={group} has tags={group_tags}\n\n")
#             cluster_data.append(dict())
#             cluster_data[-1]["tags"] = list()
#             for tag in group_tags:
#                 most_rel = most_relevant_movies(group, movie_to_group, deny_list, tag, loader.tag_counts_per_movie, loader)
#                 movies_without_url = []
#                 d = {
#                     "tag": tag,
#                     "movies": []
#                 }
#                 for movie_idx in most_rel:
#                     img_url = loader.get_image(movie_idx)
#                     mov = {
#                         "movie": loader.movie_index_to_description[movie_idx],
#                         "url": img_url,
#                         "movie_idx": movie_idx,
#                         "description": loader.movie_index_to_description[movie_idx]
#                     }
#                     if not img_url:
#                         movies_without_url.append(mov)
#                     else:
#                         d["movies"].append(mov)
#                     if len(d["movies"]) >= NUM_MOVIES_PER_TAG:
#                         print(f"Achieved: {len(d['movies'])} out of {NUM_MOVIES_PER_TAG} needed")
#                         break
                
#                 if len(d["movies"]) < NUM_MOVIES_PER_TAG:
#                     remaining = NUM_MOVIES_PER_TAG - len(d["movies"])
#                     d["movies"].extend(movies_without_url[:remaining])
                
#                 # d = {
#                 #     "tag": tag,
#                 #     "movies": [
#                 #         {
#                 #             "movie": loader.movie_index_to_description[movie_idx],
#                 #             "url": get_image(loader.links_df.loc[loader.movie_index_to_id[movie_idx]].imdbId),
#                 #             "movie_idx": movie_idx,
#                 #             "description": loader.movie_index_to_description[movie_idx]
#                 #         }
#                 #         for movie_idx in most_rel
#                 #     ]
#                 # }
#                 #deny_list.update(most_rel)
#                 deny_list.update([m["movie_idx"] for m in d["movies"]])
#                 print(f"Deny list: {deny_list}")
#                 cluster_data[-1]["tags"].append(d)

#         print(f"## Adding most relevant movies took: {time.perf_counter() - start_time}")
#         # Result is a list of clusters, each cluster being a dict (JSON object) and having a list of movies and other properties
#         print(f"Cluster data: {cluster_data}")
#         return cluster_data
#     return cluster_data


def load_data_1(elicitation_movies):
    loader = load_ml_dataset()

    # Get list of items
    start_time = time.perf_counter()
    data, extra_data = MultiObjectiveSamplingFromBucketsElicitation(
        loader.rating_matrix,
        loader.similarity_matrix,
        {
            "relevance": 2, "diversity": 2, "novelty": 2
        },
        {
            "relevance": [4, 4], "diversity": [4, 4], "novelty": [4, 4]
        }
    ).get_initial_data([int(x["movie_idx"]) for x in elicitation_movies])
    print(f"Getting initial data took: {time.perf_counter() - start_time}")
    
    #print([loader.movie_index_to_description[movie_idx] for movie_idx in data])
    #print([loader.movies_df.iloc[movie_idx].title for movie_idx in data])
    start_time = time.perf_counter()
    # res = [loader.movie_index_to_description[movie_idx] for movie_idx in data]
    res = [loader.movie_index_to_description[movie_idx] + " # " + extra for movie_idx, extra in zip(data, extra_data)]
    
    
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

def load_data_2(elicitation_movies):
    
    loader = load_ml_dataset()

    # Get list of items
    start_time = time.perf_counter()
    data = PopularitySamplingElicitation(loader.rating_matrix, n_samples=16).get_initial_data([int(x["movie_idx"]) for x in elicitation_movies])
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

def load_data_3(elicitation_movies):
    loader = load_ml_dataset()
    # Get list of items
    data = PopularitySamplingFromBucketsElicitation(loader.rating_matrix, 5, [4]*5).get_initial_data([int(x["movie_idx"]) for x in elicitation_movies])
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
    algo = als.ImplicitMF(100, iterations=50)
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
    sorted_prediction = algo.recommend(max_user + 1, n=8)
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


def calculate_weight_estimate(selected_movies, elicitation_movies):
    if not selected_movies:
        x = np.array([1.0, 1.0, 1.0])
        return x / x.sum()

    loader = load_ml_dataset()

    print(f"Selected movies = {selected_movies}")
    print(f"Elicitation movies = {elicitation_movies}")

    rel = 0
    div = 0
    nov = 0

    # relevances = []
    # diversities = []
    # novelties = []

    selected_relevances = []
    selected_diversities = []
    selected_novelties = []

    distance_matrix = 1.0 - loader.similarity_matrix

    movie_indices = [int(movie["movie_idx"]) for movie in elicitation_movies]
    diversities = distance_matrix[np.ix_(movie_indices, movie_indices)]
    #diversities = diversities[np.triu_indices(diversities.shape[0])].reshape((-1, 1)) # Train diversity on this
    diversities = (diversities.sum(axis=0) / (diversities.shape[0] - 1)).reshape(-1,1)
    print(f"Diversities = {diversities}, min={diversities.min()}, max={diversities.max()}, avg={diversities.mean()}")
    
    #relevances = loader.rating_matrix[:, movie_indices].T # Train other CDF on this
    relevances = loader.rating_matrix[:, movie_indices] # Train other CDF on this
    relevances = relevances.mean(axis=0)
    # relevances = relevances[relevances > 0.0]
    relevances = relevances.reshape((-1, 1))
    novelties = 1.0 - (loader.rating_matrix.astype(bool).sum(axis=0) / loader.rating_matrix.shape[0])
    novelties = novelties[movie_indices].reshape((-1, 1))


    for movie_idx in movie_indices:
        # relevances.append(movie["relevance"])
        # diversities.append(movie["diversity"])
        # novelties.append(movie["novelty"])
        if movie_idx in selected_movies:
            r = loader.rating_matrix[:, movie_idx]
            #r = r[r > 0.0]
            r = r.mean(axis=0)
            #r = r.mean()
            selected_relevances.append(r) # TODO try without transposition
            # selected_diversities.append(distance_matrix[movie_idx][movie_indices].mean())
            d = distance_matrix[movie_idx][movie_indices].sum() / (len(movie_indices) - 1)
            selected_diversities.append(d[d > 0].reshape(-1, 1))
            selected_novelties.append(1.0 - (loader.rating_matrix[:, movie_idx].astype(bool).sum() / loader.rating_matrix.shape[0]))
            #selected_relevances.append(movie["relevance"])
            #selected_diversities.append(movie["diversity"])
            #selected_novelties.append(movie["novelty"])
    
    # Take relevance values of all items and train CDF on it
    # Calculate CDF for relevances of all selected items and take their average
    rel_cdf = QuantileTransformer()
    #print(f"Fitting relevance cdf on relevances: {relevances.shape}, min={relevances.min()}, max={relevances.max()}, avg={relevances.mean()}, perc50={np.percentile(relevances, 50)}")
    rel_cdf.fit(relevances)
    x = np.stack(selected_relevances).reshape(-1, 1)
    #print(f"Transforming relevances={x.shape}, min={x.min()}, max={x.max()}, avg={x.mean()}, perc50={np.percentile(x, 50)}")
    rel = np.mean(rel_cdf.transform(x))
    #print(f"Selected relevances = {selected_relevances}")
    
    #print(f"Rel={rel}")

    # Take diversity values of all items shown during elicitation and train CDF on it
    # Calculate CDF for diversities of all selected items and take their average
    div_cdf = QuantileTransformer()
    div_cdf.fit(diversities)
    print(f"Selected diversities = {np.stack(selected_diversities).reshape(-1, 1)}, min={np.stack(selected_diversities).reshape(-1, 1).min()}, mean={np.stack(selected_diversities).reshape(-1, 1).mean()}")
    div = np.mean(div_cdf.transform(np.stack(selected_diversities).reshape(-1, 1)))

    # Take novelty values of all items and train CDF on it
    # Calculate CDF for novelties of all selected items and take their average
    nov_cdf = QuantileTransformer()
    nov_cdf.fit(novelties)
    nov = np.mean(nov_cdf.transform(np.stack(selected_novelties).reshape(-1, 1)))

    result = np.array([rel, div, nov])
    return result / result.sum()


@functools.lru_cache(maxsize=None)
def prepare_wrapper_once():
    loader = load_ml_dataset()


    items = np.arange(loader.rating_matrix.shape[1])
    distance_matrix = 1.0 - loader.similarity_matrix
    print(f"Rating matrix min={loader.rating_matrix.min()}, max={loader.rating_matrix.max()}, avg={loader.rating_matrix.mean()}")
    
    users_viewed_item = loader.rating_matrix.astype(bool).sum(axis=0)
    print(f"Total entries = {users_viewed_item.sum()} should equal dataframe size: {loader.ratings_df.shape}")
    # extended_rating_matrix = loader.rating_matrix.copy() # TODO prepare extended rating matrix
    
    #ratings_df = loader.ratings_df.rename(columns={"movieId": "item", "userId": "user"})

    #algo = als.ImplicitMF(100, iterations=50)
    #algo = Recommender.adapt(algo)
    #algo.fit(ratings_df)
    
    # start_time = time.perf_counter()
    # item_ids = ratings_df.item.unique()
    # for user_id in ratings_df.user.unique():
    #     user_idx = loader.user_to_user_index[user_id]
    #     res = algo.predict_for_user(user_id, item_ids)
    #     res = ((res - res.min()) / (res.max() - res.min())) * 5.0
    #     for item_id, rating in res.iteritems():
    #         item_idx = loader.movie_id_to_index[item_id]
    #         if extended_rating_matrix[user_idx, item_idx] == 0.0:
    #             extended_rating_matrix[user_idx, item_idx] = rating
    # print(f"Took: {time.perf_counter() - start_time}")

    return loader, items, distance_matrix, users_viewed_item#, algo, ratings_df

def enrich_results(top_k, loader):
    top_k_description = [loader.movie_index_to_description[movie_idx] for movie_idx in top_k]
    top_k_url = [loader.get_image(movie_idx) for movie_idx in top_k]
    return [{"movie": movie, "url": url, "movie_idx": str(movie_idx)} for movie, url, movie_idx in zip(top_k_description, top_k_url, top_k)]

def prepare_wrapper(selected_movies, model, mandate_allocation_factory, obj_weights, filter_out_movies = []):
    loader, items, distance_matrix, users_viewed_item = prepare_wrapper_once()

    max_user = loader.ratings_df.userId.max()
    model, train = prepare_tf_model(loader)
    new_user = tf.constant(str(max_user + 1))
    def data_gen():
        for x in selected_movies:
            yield {
                "movie_title": tf.constant(loader.movies_df.loc[x].title),
                "user_id": new_user,
            }
    ratings2 = tf.data.Dataset.from_generator(data_gen, output_signature={
        "movie_title": tf.TensorSpec(shape=(), dtype=tf.string),
        "user_id": tf.TensorSpec(shape=(), dtype=tf.string)
    })
    scores, x = model.predict_all_unseen(new_user, ratings2, n_items=items.size) #model.predict_for_user(new_user, ratings2, k=2000)
    scores, x = tf.squeeze(scores).numpy(), tf.squeeze(x).numpy()
    #print(f"### Recommendations = {x[:10]}")
    #print(f"### Scores shape={scores[:10]}")
    #print(f"Min score={scores.min()}, max={scores.max()}, mean={scores.mean()}")
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    top_k = [loader.movie_id_to_index[loader.movies_df[loader.movies_df.title == x.decode("UTF-8")].movieId.values[0]] for x in x]
    
    rating_vector = np.zeros((items.size, ), dtype=np.float32)
    rating_vector[top_k] = scores # Set the predicted scores
    rating_vector[selected_movies] = 1.0 # Set selected movies to 1.0
    rating_vector = np.expand_dims(rating_vector, axis=0) # Convert it to 1xN rating matrix
    print(f"Min rating vector={rating_vector.min()}, max rating vector = {rating_vector.max()}, avg={rating_vector.mean()}")
    #print(f"Rating_vector = {rating_vector}")
    extended_rating_matrix = rating_vector #loader.rating_matrix[:1] #rating_vector
    #print(f"### Extended rating matrix shape = {extended_rating_matrix.shape}, rating_vector shape={rating_vector.shape}")
    
    normalization_factory = standardization
    cache_dir = "."

    mandate_allocation = mandate_allocation_factory(obj_weights, -1e6)
    
    unseen_items_mask = np.ones(extended_rating_matrix.shape, dtype=np.bool8)
    #unseen_items_mask[loader.rating_matrix > 0.0] = 0 # Mask out already seem items
    if filter_out_movies:
        unseen_items_mask[:, np.array(filter_out_movies)] = 0 # Mask out the items

    # Reduce for single user
    # extended_rating_matrix = extended_rating_matrix[:1]
    # unseen_items_mask = unseen_items_mask[:1]


    discount_sequences = [[1.0] * 10, [1.0] * 10, [1.0] * 10]

    return loader, RLPropWrapper(items, extended_rating_matrix, distance_matrix, users_viewed_item, normalization_factory, mandate_allocation, unseen_items_mask, cache_dir, discount_sequences)

def rlprop(selected_movies, model, weights, filter_out_movies = []):
    obj_weights = weights
    obj_weights /= obj_weights.sum()
    

    loader, wrapper = prepare_wrapper(selected_movies, model, exactly_proportional_fuzzy_dhondt_2, obj_weights, filter_out_movies)
    wrapper.init()
    x = wrapper(10)

    return enrich_results(x[0], loader)

def weighted_average(selected_movies, model, weights, filter_out_movies = []):
    obj_weights = weights
    obj_weights /= obj_weights.sum()
    
    loader, wrapper = prepare_wrapper(selected_movies, model, weighted_average_strategy, obj_weights, filter_out_movies)
    wrapper.init()
    x = wrapper(10)

    return enrich_results(x[0], loader)


def recommend_2_3(selected_movies, filter_out_movies = [], return_model = False):
    loader = load_ml_dataset()

    # algo_als = als.BiasedMF(10, iterations=5)
    #algo = als.ImplicitMF(100, iterations=50)
    #algo = Recommender.adapt(algo)

    max_user = loader.ratings_df.userId.max()

    ################ TF specific ################
    model, train = prepare_tf_model(loader)

    
    
    print(f"Selected: {selected_movies}\n\n")
    new_user = tf.constant(str(max_user + 1))
    def data_gen():
        for x in selected_movies:
            yield {
                "movie_title": tf.constant(loader.movies_df.loc[x].title),
                "user_id": new_user,
            }
    ratings2 = tf.data.Dataset.from_generator(data_gen, output_signature={
        "movie_title": tf.TensorSpec(shape=(), dtype=tf.string),
        "user_id": tf.TensorSpec(shape=(), dtype=tf.string)
    })


    filter_out_movies_titles = [bytes(loader.movies_df.loc[x].title, "UTF-8") for x in filter_out_movies]


    for x in ratings2:
        print(f"x={x['movie_title'].numpy()} Is in unique: {x['movie_title'].numpy() in np.unique(np.concatenate(list(model.movies.batch(1000))))}")
    print(f"Uniques: {np.unique(np.concatenate(list(model.movies.batch(1000))))}")

    print(f"Predictions are: {model.predict_for_user(new_user, ratings2)}")
    print(f"Predictions are: {model.predict_for_user(new_user, ratings2)}")

    # Finetune
    start_time = time.perf_counter()
    model.fit(ratings2.concatenate(train.take(100)).batch(256), epochs=2)
    predictions = tf.squeeze(model.predict_for_user(new_user, ratings2, filter_out_movies_titles)).numpy()
    print(f"Predictions (AFTER FINETUNING) are: {predictions}")
    print(f"Fine tuning took: {time.perf_counter() - start_time}")

    
    top_k = [loader.movie_id_to_index[loader.movies_df[loader.movies_df.title == x.decode("UTF-8")].movieId.values[0]] for x in predictions]
    print(f"Prediction indices={top_k}")

    # model({"user_id": np.array(["0"]), "movie_title": np.array(["unknown"])})
    # model.summary()
    # cached_train2 = ratings2.shuffle(100_000).batch(8192).cache()
    # # Freeze movie model
    # model.ranking_model.movie_embeddings.trainable = False
    # model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01))
    # # Finetune
    # model.fit(cached_train2, epochs=10)

    # print(f"@@ Predicting for user: {str(max_user + 1)}")
    # predicted_scores = model({
    #     "user_id": tf.repeat(tf.constant(str(max_user + 1)), repeats=model.unique_movie_titles.shape[0]),
    #     "movie_title": model.unique_movie_titles
    # })
    # print(f"Predicted = {predicted_scores.numpy()}")
    # _, top_k = tf.math.top_k(tf.squeeze(predicted_scores), k=10)
    # top_k = top_k.numpy()
    # print(f"Top k = {top_k}")
    
    if return_model:
        return enrich_results(top_k, loader), model

    return enrich_results(top_k, loader)

    
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
    sorted_prediction = algo.recommend(max_user + 1, n=8)
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


def search_for_movie(attrib, pattern):

    # Map to 
    if attrib == "movie":
        attrib = "title"

    loader = load_ml_dataset()
    found_movies = loader.movies_df[loader.movies_df.title.str.contains(pattern, case=False)]
    
    movie_indices = [loader.movie_id_to_index[movie_id] for movie_id in found_movies.movieId.values]
    res_url = [loader.get_image(movie_idx) for movie_idx in movie_indices]
    result = [{"movie": movie, "url": url, "movie_idx": str(movie_idx)} for movie, url, movie_idx in zip(found_movies.title.values, res_url, movie_indices)]

    return result


if __name__ == "__main__":
    loader = load_ml_dataset()


    items = np.arange(loader.rating_matrix.shape[1])
    distance_matrix = 1.0 - loader.similarity_matrix
    print(f"Rating matrix min={loader.rating_matrix.min()}, max={loader.rating_matrix.max()}, avg={loader.rating_matrix.mean()}")
    
    users_viewed_item = loader.rating_matrix.astype(bool).sum(axis=0)
    print(f"Total entries = {users_viewed_item.sum()} should equal dataframe size: {loader.ratings_df.shape}")
    extended_rating_matrix = loader.rating_matrix.copy() # TODO prepare extended rating matrix
    
    ratings_df = loader.ratings_df.rename(columns={"movieId": "item", "userId": "user"})

    algo = als.ImplicitMF(100, iterations=50)
    algo = Recommender.adapt(algo)
    algo.fit(ratings_df)
    
    start_time = time.perf_counter()
    item_ids = ratings_df.item.unique()
    for user_id in ratings_df.user.unique():
        user_idx = loader.user_to_user_index[user_id]
        res = algo.predict_for_user(user_id, item_ids)
        res = ((res - res.min()) / (res.max() - res.min())) * 5.0
        for item_id, rating in res.iteritems():
            item_idx = loader.movie_id_to_index[item_id]
            if extended_rating_matrix[user_idx, item_idx] == 0.0:
                extended_rating_matrix[user_idx, item_idx] = rating
    print(f"Took: {time.perf_counter() - start_time}")
    
    
    normalization_factory = identity
    cache_dir = "."

    obj_weights = np.array([1.0, 1.0, 1.0])
    obj_weights /= obj_weights.sum()
    mandate_allocation = exactly_proportional_fuzzy_dhondt_2(obj_weights, -1e6)
    
    unseen_items_mask = np.ones(loader.rating_matrix.shape, dtype=np.bool8)
    unseen_items_mask[loader.rating_matrix > 0.0] = 0 # Mask out already seem items


    discount_sequences = [[1.0] * 10, [1.0] * 10, [1.0] * 10]

    wrapper = RLPropWrapper(items, extended_rating_matrix, distance_matrix, users_viewed_item, normalization_factory, mandate_allocation, unseen_items_mask, cache_dir, discount_sequences)
    wrapper.init()
    x = wrapper(10)
    print(x)




    # x = MultiObjectiveSamplingFromBucketsElicitation(loader.rating_matrix, loader.similarity_matrix, {"relevance": 3, "diversity": 2, "novelty": 1}, {"relevance": [4, 4, 4], "diversity": [6, 4], "novelty": [5]})
    # res = x.get_initial_data()
    # print(f"Got initial data = {res}")
    #recommend_2_3([1225, 1244, 927, 1081, 929, 1071, 1269, 1402, 838, 1151])
    
    # loader = load_ml_dataset()
    # pass