import tensorflow as tf
import tensorflow_recommenders as tfrs

from typing import Dict, Text

# class RankingModel(tf.keras.Model):

#   def __init__(self, unique_user_ids, unique_movie_titles):
#     super().__init__()
#     embedding_dimension = 32

#     # Compute embeddings for users.
#     self.user_embeddings = tf.keras.Sequential([
#       tf.keras.layers.StringLookup(
#         vocabulary=unique_user_ids, mask_token=None),
#       tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
#     ])

#     # Compute embeddings for movies.
#     self.movie_embeddings = tf.keras.Sequential([
#       tf.keras.layers.StringLookup(
#         vocabulary=unique_movie_titles, mask_token=None),
#       tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
#     ])

#     # Compute predictions.
#     self.ratings = tf.keras.Sequential([
#       # Learn multiple dense layers.
#       tf.keras.layers.Dense(256, activation="relu"),
#       tf.keras.layers.Dense(64, activation="relu"),
#       # Make rating predictions in the final layer.
#       tf.keras.layers.Dense(1)
#   ])

#   def call(self, inputs):

#     user_id, movie_title = inputs

#     user_embedding = self.user_embeddings(user_id)
#     movie_embedding = self.movie_embeddings(movie_title)

#     return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))


# class MovielensModel(tfrs.models.Model):

#   def __init__(self, unique_user_ids, unique_movie_titles):
#     super().__init__()
#     self.ranking_model: tf.keras.Model = RankingModel(unique_user_ids, unique_movie_titles)
#     self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
#       loss = tf.keras.losses.MeanSquaredError(),
#       metrics=[tf.keras.metrics.RootMeanSquaredError()]
#     )
#     self.unique_movie_titles = unique_movie_titles

#   def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
#     return self.ranking_model(
#         (features["user_id"], features["movie_title"]))

#   def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
#     labels = features.pop("user_rating")

#     rating_predictions = self(features)

#     # The task computes the loss and the metrics.
#     return self.task(labels=labels, predictions=rating_predictions)


class MovielensRetrievalModel(tfrs.models.Model):

    def __init__(self, user_model, movie_model, task, movies):
        super().__init__()
        self.movie_model: tf.keras.Model = movie_model
        self.user_model: tf.keras.Model = user_model
        self.task: tf.keras.layers.Layer = task
        self.movies = movies

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["user_id"])
        # And pick out the movie features and pass them into the movie model,
        # getting embeddings back.
        positive_movie_embeddings = self.movie_model(features["movie_title"])

        # The task computes the loss and the metrics.
        return self.task(user_embeddings, positive_movie_embeddings, compute_metrics=False)

    def predict_for_user(self, user, new_ratings, filter_out_movie_titles=[], k=10):

        # Generate prediction
        seen_movies = set()
        for x in new_ratings:
            seen_movies.add(x["movie_title"].numpy())

        print(f"Seen movies={seen_movies}")
        seen_movies.update(filter_out_movie_titles)
        print(f"Seen movies after extension with {filter_out_movie_titles} got ={seen_movies}")
        print(f"Num seen movies = {len(seen_movies)}")

        unseen_movies = self._users_unseen_movies(self.movies, seen_movies)

        # Create a model that takes in raw query features, and
        index = tfrs.layers.factorized_top_k.BruteForce(self.user_model, k=k)
        # recommends movies out of the entire movies dataset.
        index.index_from_dataset(
            #tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model)))
            tf.data.Dataset.zip((unseen_movies.batch(100), unseen_movies.batch(100).map(self.movie_model)))
        )

        # Get recommendations.
        _, titles = index(tf.expand_dims(user, axis=0))


        return titles[:, :k]

    def predict_all_unseen(self, user, new_ratings, n_items, filter_out_movie_titles=[]):
        # Generate prediction
        seen_movies = set()
        for x in new_ratings:
            seen_movies.add(x["movie_title"].numpy())
        seen_movies.update(filter_out_movie_titles)
        unseen_movies = self._users_unseen_movies(self.movies, seen_movies)

        k = n_items - len(seen_movies)

        # Create a model that takes in raw query features, and
        index = tfrs.layers.factorized_top_k.BruteForce(self.user_model, k=k)
        # recommends movies out of the entire movies dataset.
        index.index_from_dataset(
            #tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model)))
            tf.data.Dataset.zip((unseen_movies.batch(100), unseen_movies.batch(100).map(self.movie_model)))
        )

        # Get recommendations.
        scores, titles = index(tf.expand_dims(user, axis=0))

        return scores[:, :k], titles[:, :k]

    # User's unseen movies
    def _users_unseen_movies(self, movies, users_seen_movies):
        def fnc_impl(x):
            return x.numpy() not in users_seen_movies
        def fnc(x):
            return tf.py_function(fnc_impl, [x], tf.bool)
        return movies.filter(fnc)

    def _filter_user(self, x, user):
        return x["user_id"] != user

def get_model_25m(unique_user_ids, unique_movie_titles, movies):
    embedding_dimension = 32

    user_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
        vocabulary=unique_user_ids, mask_token=None),
        # We add an additional embedding to account for unknown tokens.
        tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
    ])

    movie_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_movie_titles, mask_token=None),
        tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
    ])

    metrics = tfrs.metrics.FactorizedTopK(
        candidates=movies.batch(128).map(movie_model)
    )

    task = tfrs.tasks.Retrieval(
        metrics=metrics,
        #batch_metrics=[tfr.keras.metrics.NDCGMetric()]
    )

    model = MovielensRetrievalModel(user_model, movie_model, task, movies)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

    return model


