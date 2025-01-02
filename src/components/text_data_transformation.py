import os
from dataclasses import dataclass
import joblib as jl
import regex as re


import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.logger import logger
from src.constants import num_columns


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file :str = os.path.join("artifacts", "text_preprocessor.joblib")
    transformed_train_file : str = os.path.join("artifacts", "transformed_train_data.csv")
    transformed_test_file : str = os.path.join("artifacts", "transformed_test_data.csv")


class remove_stop_words(TransformerMixin, BaseEstimator):

    """"
    This class contains necessary methods to remove stop words, tokenize, lowercase and remove non-alphabetic characters
    """

    def __init__(self):
        try:
            logger.info("Getting set of stopwords from NLTK for English language")
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logger.info(f"Error loading stopwords from NLTK library: {e}")
            raise e

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.X_ = X.copy()
        try:
            logger.info("Transforming all the letters into lowercase")
            self.X_ = self.X_.map(lambda x: x.lower())

            logger.info("Tokenizing and removing stopwords")
            pattern = re.compile(r'[^a-zA-Z\s]+')  # Allow spaces
            self.X_ = self.X_.map(
                lambda x: ' '.join(
                    [pattern.sub('', word) for word in x.split() if word not in self.stop_words ]
                )
            )
            logger.info("Text cleaning and stopword removal successful")
        except Exception as e:
            logger.info(f"Error in preprocessing the text: {e}")
            raise e
        return self.X_



class lemmatization(TransformerMixin, BaseEstimator):
    """
    This class contains methods for cleaned text after removing stop words and tokenization to lemmatize the words
    """

    def __init__(self):
        try:
            logger.info("Instantiating WordNetLemmatizer from NLTK")
            self.lemmatizer = WordNetLemmatizer()
        except Exception as e:
            logger.info(f"Error initializing the lemmatizer: {e}")
            raise e

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.X_ = X.copy()
        try:
            logger.info("Lemmatizing the words")
            self.X_ = self.X_.map(
                lambda x: ' '.join([self.lemmatizer.lemmatize(word) for word in x.split()])
            )
            logger.info("Lemmatization successful")
        except Exception as e:
            logger.info(f"Error in lemmatization: {e}")
            raise e
        return self.X_

    


class DataTransformation:

    def __init__ (self):
        try:
            logger.info(f"Initiating the DataTransformation")
            self.data_transformation_config = DataTransformationConfig()
            pass
        
        except Exception as e:
            logger.info(f"Error initiating the DataTransformation : {e}")
            raise e
        

    def get_transformer_object(self):
        try:

            text_pipeline = Pipeline(
                [
                ("clean and preprocess", remove_stop_words()),
                ("lemmatize the words", lemmatization())
                ]
            )
            
            logger.info("Text pipeline creation successful")
            
            return text_pipeline
        except Exception as e:
            raise e
    
    def initiate_data_transformation(self, train_data_path, test_data_path):

        try:
            logger.info(f"Initiating data transformation pipeline")
            train_data = pd.read_csv(train_data_path)
            train_data.dropna(inplace=True, axis=0)
            train_data.drop(columns=num_columns, axis=1, inplace=True)

            test_data = pd.read_csv(test_data_path)
            test_data.dropna(inplace=True, axis=0)
            test_data.drop(columns=num_columns, axis=1, inplace=True)
            


            preprocessing_obj = self.get_transformer_object()

            logger.info("Fitting and transforming training data")

            input_feature_arr = preprocessing_obj.fit_transform(train_data)


            logger.info("Transforming test data")

            input_test_arr = preprocessing_obj.transform(test_data)

            logger.info(f"Transforming test data")

            jl.dump(preprocessing_obj, self.data_transformation_config.preprocessor_obj_file)
            logger.info(f"Saved fitted preprocessor to {self.data_transformation_config.preprocessor_obj_file}")



            logger.info(f"saving transformed train data and test data at {self.data_transformation_config.transformed_train_file} and {self.data_transformation_config.transformed_test_file} respectively")

            input_feature_arr.to_csv(self.data_transformation_config.transformed_train_file, index=False)

            input_test_arr.to_csv(self.data_transformation_config.transformed_test_file, index=False)

            logger.info(f"Returning the transformed input train feature as an array and  transfomed test feature as array respectively")

            return input_feature_arr, input_test_arr
        except Exception as e:
            logger.info(f"Error in initiating the data transformation pipeline {e}")
            raise e

