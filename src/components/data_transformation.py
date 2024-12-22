import os
from pathlib import Path
from dataclasses import dataclass
import joblib as jl
import regex as re


import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import KeyedVectors

from src.utils import combine_main_and_subarrays
from src.logger import logger
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.constants import num_columns,text_columns
from src.db_paths import fast_text_model


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file :str = os.path.join("artifacts", "preprocessor.joblib")
    transformed_train_file : str = os.path.join("artifacts", "transformed_train_data.csv")
    transformed_test_file : str = os.path.join("artifacts", "transformed_test_data.csv")

@dataclass
class DataColumns:
    Numerical_Columns = num_columns
    Text_Columns = text_columns

class remove_stop_words(TransformerMixin, BaseEstimator):

    def __init__(self):
        
        try:
            logger.info("getting set of stopwords from NLTK from English language")
            self.stop_words = set(stopwords.words('english'))
            pass
        except Exception as e:
            logger.info(f"Error loading set of stopwords from NLTK library: {e}")
            raise e
        

    def fit(self, X):
        return self


    def transform(self, X):
        self.X_ = X.copy()
        try:
            logger.info(f"Transforming all the letters into lowercase")
            self.X_ = self.X_.apply(lambda x: x.str.lower())
            logger.info(f"Transforming all letters into lowercase is successful")

            logger.info(f"Splitting the sentences into words (Tokenization)")
            self.X_ = self.X_.apply(lambda x: x.str.split())
            logger.info(f"Splitting the sentences into words is successful (Tokenization)")

            pattern = re.compile(r'[^a-zA-Z\s]+')  # Allow spaces
            logger.info(f"Removing punctuations from text")
            self.X_ = self.X_.map(lambda x: [pattern.sub('', word) for word in x])
            logger.info(f"Removing punctuations from text successful")


            logger.info(f"Removing the stopwords from the tokenized words")
            self.X_ = self.X_.map(lambda x: [word for word in x if not word in self.stop_words ])
            logger.info(f"Removing the stopwords from the tokenized words is successful")
        except Exception as e:
            logger.info(f"Error in pre-processing the text: {e}")
            raise e
        
        return self.X_


class lemmatization(TransformerMixin, BaseEstimator):

    def __init__ (self):
        try:
            logger.info(f"WordNetLemmatizer from NLTK library is instantiated")
            self.lemmatizer = WordNetLemmatizer()
            pass
        except Exception as e:
            logger.info(f"Error initiating the lemmatizer from NLTK library: {e}")
            raise e
        
    
    def fit(self, X):
        return self
    
    def transform(self, X):
        self.X_ = X.copy()
        try:
            logger.info(f"Initiating the lemmatization of the remaining words after stopword removal")
            self.X_ = self.X_.map(lambda x: [self.lemmatizer.lemmatize(word) for word in x])
            logger.info(f"Lemmatization successful")
        except Exception as e:
            logger.info(f"Error in lemmatization: {e}")
            raise e
        return self.X_
    
class make_embeddings(TransformerMixin, BaseEstimator):

    def __init__(self):
        try:
            logger.info("Loading Gensim Word2Vec model for embedding generation")
            self.word2vec = KeyedVectors.load_word2vec_format("ft_reviews_vectors.bin", binary=True)  # Update path as needed
            pass
        except Exception as e:
            logger.info(f"Error loading Gensim Word2Vec model: {e}")
            raise e

    def fit(self, X):

        return self

    def transform(self, X):
        self.X_ = X.copy()
        try:
            logger.info("Generating embeddings for the text data")
            
            # Compute embeddings using Gensim's get_mean_vector method
            embeddings = self.X_.map(lambda words: self.word2vec.get_mean_vector(words, ignore_missing=True))

            logger.info("Successfully generated embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Error during transformation: {e}")
            raise e

class DataTransformation():

    def __init__ (self):
        try:
            logger.info(f"Initiating the DataTransformation")
            self.data_transformation_config = DataTransformationConfig()
            self.data_columns = DataColumns()
            pass
        
        except Exception as e:
            logger.info(f"Error initiating the DataTransformation : {e}")
            raise e
        

    def get_transformer_object(self):
        try:
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
                ]
            )

            text_pipeline = Pipeline([
                ("clean and preprocess", remove_stop_words()),
                ("lemmatize the words", lemmatization()),
                ("Create Embeddings", make_embeddings())
            ])
            
            logger.info("Numerical and text pipelines created")
            
            preprocessor = ColumnTransformer([
                ("Numerical_Pipeline", num_pipeline, self.data_columns.Numerical_Columns),
                ("Text_Pipeline", text_pipeline, self.data_columns.Text_Columns)
            ])

            return preprocessor  
        except Exception as e:
            raise e
    
    def initiate_data_transformation(self, train_data_path, test_data_path):

        try:
            logger.info(f"Initiatig data transformation pipeline")
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)
            train_data.dropna(inplace=True, axis=0)
            test_data.dropna(inplace=True, axis=0)
            


            preprocessing_obj = self.get_transformer_object()
            logger.info("Fitting and transforming training data")
            input_feature_arr = preprocessing_obj.fit_transform(train_data)
            logger.info("Transforming test data")
            input_test_arr = preprocessing_obj.transform(test_data)


            input_feature_arr = combine_main_and_subarrays(input_feature_arr)
            logger.info(f"Transforming test data")
            input_test_arr = combine_main_and_subarrays(input_test_arr)


            jl.dump(preprocessing_obj, self.data_transformation_config.preprocessor_obj_file)
            logger.info(f"Saved fitted preprocessor to {self.data_transformation_config.preprocessor_obj_file}")

            

            logger.info(f"saving transformed train data and test data at {self.data_transformation_config.transformed_train_file} and {self.data_transformation_config.transformed_test_file} respectively")

            feature_df = pd.DataFrame(input_feature_arr.astype('float32'), columns=[f"Column{i}" for i in range(1,27)])
            # Save as CSV
            feature_df.to_csv(self.data_transformation_config.transformed_train_file, index=False)

            test_df = pd.DataFrame(input_test_arr.astype('float32'), columns=[f"Column{i}" for i in range(1,27)])
            # Save as CSV
            test_df.to_csv(self.data_transformation_config.transformed_test_file, index=False)


            logger.info(f"Returning the input train feature as an array and test feature as array respectively")
            return input_feature_arr.astype('float32'), input_test_arr.astype('float32')
        except Exception as e:
            logger.info(f"Error in initiating the data transformation pipeline {e}")
            raise e

