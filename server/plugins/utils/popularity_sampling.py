import os
import pandas as pd
import numpy as np

MIN_RATINGS_PER_USER = 500
MIN_RATINGS_PER_MOVIE = 500


class PopularitySamplingFromBucketsElicitation:
    def __init__(self, rating_matrix, n_buckets, n_samples_per_bucket, k=1.0):
        assert n_buckets == len(n_samples_per_bucket)
        self.rating_matrix = rating_matrix
        self.n_buckets = n_buckets
        self.n_samples_per_bucket = n_samples_per_bucket
        self.k = k

    def _calculate_item_popularities(self, rating_matrix):
        #  Umocnit na K
        return np.power(np.sum(rating_matrix > 0.0, axis=0) / rating_matrix.shape[0], self.k)

    def get_initial_data(self):
        popularities = self._calculate_item_popularities(self.rating_matrix)
        indices = np.argsort(-popularities)
        sorted_popularities = popularities[indices]
        sorted_items = np.arange(popularities.shape[0])[indices]
        assert sorted_popularities.ndim == sorted_items.ndim

        n_items_total = sum(self.n_samples_per_bucket)
        result = np.zeros((n_items_total, ), dtype=np.int32)

        offset = 0
        for items_bucket, popularities_bucket, n_samples in zip(
            np.array_split(sorted_items, self.n_buckets),
            np.array_split(sorted_popularities, self.n_buckets),
            self.n_samples_per_bucket
        ):
            samples = np.random.choice(items_bucket, size=n_samples, p=popularities_bucket/popularities_bucket.sum(), replace=False)
            result[offset:offset+n_samples] = samples
            offset += n_samples
            
        
        np.random.shuffle(result)
        return result

# Popularity-sampling based implementation of preference elicitation
class PopularitySamplingElicitation:
    
    def __init__(self, rating_matrix, n_samples=10, k=1.0):
        self.rating_matrix = rating_matrix
        self.n_samples = n_samples
        self.k = k

    def _calculate_item_popularities(self, rating_matrix):
        return np.power(np.sum(rating_matrix > 0.0, axis=0) / rating_matrix.shape[0], self.k)


    # Returns data to be shown to the user
    def get_initial_data(self):
        popularities = self._calculate_item_popularities(self.rating_matrix)
        p_popularities = popularities / popularities.sum()
        print(f"Popularities = {popularities}")
        print(f"p_popularities = {p_popularities}")
        s = np.random.choice(np.arange(self.rating_matrix.shape[1]), p=p_popularities, size=self.n_samples, replace=False)
        print(f"Sample: {s}")
        print(f"RM shape: {self.rating_matrix.shape}")
        print(f"Sample items: {popularities[s]}")
        np.random.shuffle(s)
        return s

    # Based on user selection of initial data, return vector with ratings of all the items
    def get_preferences(self, user_selection):
        pass
