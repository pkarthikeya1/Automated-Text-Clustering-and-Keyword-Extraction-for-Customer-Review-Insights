import warnings
import os
warnings.filterwarnings("ignore")

from dataclasses import dataclass

import pandas as pd
import numpy as np
import joblib as jl

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.cluster import DBSCAN
from sklearn.pipeline import Pipeline
import umap

from src.logger import logger
from src.components.data_transformation import DataTransformationConfig


@dataclass
class ModellingConfig:
    model_object : str = os.path.join('artifacts','cluster_model.joblib')


class Reduce_Dimesionality(TransformerMixin, BaseEstimator):

    def __init__ (self):

        logger.info(f"Initiating the UMAP model for dimensionality reduction")
        logger.info(f"The n_neighbours =100, min_dist =0.5, metric=cosine")

        try:
            self.reducer = umap.UMAP(n_neighbors = 100, min_dist = 0.5, metric='cosine', n_jobs=-1, random_state=42, )
            pass
        except Exception as e:
            logger.info(f"Error initializing the UMAP: {e}")
            raise e
        

    def fit(self, X, y=None):

        try:
            logger.info(f"Fitting the UMAP to the given data")
            X_ = self.reducer.fit(X)
            return X_
        
        except Exception as e:
            logger.info(f"Error fitting the data with UMP : {e}")
            raise e
    


    def transform(self, X, y=None):
        try:
            logger.info(f"Transforming the given data using fitted UMAP model")
            X_ = self.reducer.transform(X)
            logger.info(f"Output shape after UMAP transform: {X_.shape}")
            return X_  
        except Exception as e:
            logger.error(f"Error transforming the data using the UMAP model: {e}")
            raise e


class ClusterData(TransformerMixin, BaseEstimator):
    def __init__(self):
        logger.info(f"Initializing the DBSCAN clustering model")
        try:
            self.dbscan = DBSCAN(eps=0.6, min_samples=100, n_jobs=-1)
            pass
        except Exception as e:
            logger.info(f"Error initializing the DBSCAN clustering model: {e}")
            raise e

    def fit(self, X, y=None):
        try:
            logger.info(f"Fitting the data to the DBSCAN cluster model")
            self.dbscan.fit(X)
            return self
        except Exception as e:
            logger.info(f"Error in fitting the data to the DBSCAN cluster model: {e}")
            raise e

    def transform(self, X, y=None):
        try:
            logger.info(f"Transforming the data using the DBSCAN clustering model")
            if X.ndim == 1:
                X = X.reshape(-1, 1)  # Ensure input is 2D
            labels = self.dbscan.fit_predict(X)
            logger.info(f"Successfully assigned clusters using DBSCAN")
            return labels.reshape(-1, 1)  # Ensure output is a 2D array
        except Exception as e:
            logger.info(f"Error in transforming the data with the DBSCAN model: {e}")
            raise e

class Cluster_Modelling:

    def __init__(self):
        self.data_trans_config = DataTransformationConfig()
        self.model_config = ModellingConfig()
        pass

    def get_cluster_model(self):

        clustering_pipeline = Pipeline(
            [
            ('Dimesionality Reduction UMAP', Reduce_Dimesionality()),
            ('Cluster Data', ClusterData())

            ]
        )
        return clustering_pipeline
    
    def initiate_clustering(self):
        try:
            logger.info(f"Initialized clustering model")
            cluster = self.get_cluster_model()

            logger.info(f"Fitting the training data to the clustering model")
            tr_data = pd.read_csv(self.data_trans_config.transformed_train_file)

            train_cluster_labels = cluster.fit_transform(tr_data.values)

            logger.info(f"saving fitted cluster model at {self.model_config.model_object}")
            jl.dump(cluster, self.model_config.model_object)

            logger.info(f"Transforming the test data to")
            ts_data = pd.read_csv(self.data_trans_config.transformed_test_file)
            test_cluster_labels = cluster.transform(ts_data.values)
    

            logger.info(f" Returning cluster labes of train data and test resectively")
            return train_cluster_labels, test_cluster_labels
        except Exception as e:
            logger.info(f" Error in initiating the clustering pipeline : {e}")
            raise e