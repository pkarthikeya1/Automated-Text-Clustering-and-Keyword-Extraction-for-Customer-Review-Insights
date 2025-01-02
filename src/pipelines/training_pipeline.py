import warnings
import os
from dataclasses import dataclass


import pandas as pd
import joblib as jl


from sklearn.preprocessing import FunctionTransformer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from gensim.models import KeyedVectors


from src.logger import logger
from src.components.data_ingestion import DataIngestionConfig
from src.components.data_transformation import DataTransformationConfig
from src.db_paths import fast_text_model

# Suppress warnings
warnings.filterwarnings("ignore")


@dataclass
class ModellingConfig:
    model_object: str = os.path.join('artifacts', 'cluster_model.joblib')
    clustered_train_data_path: str = os.path.join('artifacts', 'clustered_train_data.csv')
    clustered_test_data_path: str = os.path.join('artifacts', 'clustered_test_data.csv')

class MakeEmbeddings(TransformerMixin, BaseEstimator):
    
    def __init__(self):
        logger.info("Loading Gensim Word2Vec model for embedding generation")
        try:
            self.word2vec = KeyedVectors.load_word2vec_format(fast_text_model, binary=True)
        except Exception as e:
            logger.error(f"Error loading Word2Vec model: {e}")
            raise e

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            logger.info("Generating embeddings for the text data")
            embeddings = X.iloc[:, 0].apply(
                lambda words: self.word2vec.get_mean_vector(words, ignore_missing=True)
            )
            return pd.DataFrame(embeddings.tolist())
        except Exception as e:
            logger.error(f"Error during embedding generation: {e}")
            raise e


class ClusterData(TransformerMixin, BaseEstimator):
    def __init__(self):
        logger.info("Initializing K-Means clustering model")
        try:
            self.kmeans = KMeans(n_clusters=25, random_state=42)
        except Exception as e:
            logger.error(f"Error initializing K-Means model: {e}")
            raise e

    def fit(self, X, y=None):
        try:
            logger.info("Fitting the K-Means model")
            self.kmeans.fit(X)
            return self
        except Exception as e:
            logger.error(f"Error during K-Means fitting: {e}")
            raise e

    def transform(self, X, y=None):
        try:
            logger.info("Predicting cluster labels")
            return self.kmeans.predict(X)
        except Exception as e:
            logger.error(f"Error during K-Means prediction: {e}")
            raise e


class ClusterModelling:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_trans_config = DataTransformationConfig()
        self.model_config = ModellingConfig()

    def get_cluster_model(self):
        
        return Pipeline([
            ('Make Embeddings', MakeEmbeddings()),
            ('Cluster Data', ClusterData())
        ])

    def initiate_clustering(self, train_data_path, test_data_path):
        try:
            logger.info("Initializing clustering pipeline")
            pipeline = self.get_cluster_model()

            logger.info("Loading train and test datasets")
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)

            logger.info("Fitting and transforming train data")
            train_cluster_labels = pipeline.fit_transform(train_data)

            logger.info(f"Saving trained model at {self.model_config.model_object}")
            jl.dump(pipeline, self.model_config.model_object)

            logger.info("Transforming test data")
            test_cluster_labels = pipeline.transform(test_data)

            logger.info("Attaching cluster labels to datasets")
            train_data['cluster_id'] = train_cluster_labels
            test_data['cluster_id'] = test_cluster_labels

            train_data.to_csv(self.model_config.clustered_train_data_path, index=False)
            test_data.to_csv(self.model_config.clustered_test_data_path, index=False)

            logger.info("Clustering pipeline completed successfully")
            return train_cluster_labels, test_cluster_labels
        except Exception as e:
            logger.error(f"Error during clustering pipeline: {e}")
            raise e
