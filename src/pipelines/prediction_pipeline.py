import pandas as pd
import joblib
import os
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):

        try:
            model_path=os.path.join("artifacts","cluster_model.joblib")
            preprocessor_path=os.path.join('artifacts','text_preprocessor.joblib')

            logging.info("Loading model...")
            model=joblib.load(model_path)
            logging.info("Model loaded successfully.")

            logging.info("Loading preprocessor...")
            preprocessor=joblib.load(preprocessor_path)

            logging.info("Preprocessor loaded successfully.")
            data_scaled=preprocessor.transform(features)
            preds=model.transform(data_scaled)
            return preds
    
        except Exception as e:
            logging.error(f"Error loading artifacts: {e}")
            raise e


class CustomData:
    
    def __init__(  self,
        review_full:str):
        
        self.review_full = review_full

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age": [self.review_full]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise e