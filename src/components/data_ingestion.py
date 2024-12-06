import os
import sqlite3
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

from src.db_paths import table_name, db_path, query
from src.logger import logger


@dataclass
class DataBaseConfig:
    database_path:str = db_path
    table:str = table_name
    query:str = query


class DataBaseHandler:

    def __init__(self):
        self.DataBaseConn = DataBaseConfig()
        
    def DataFrameExraction(self):
        try:
            logger.info("Establising Connection With SQL Database")
            self.conn = sqlite3.connect(self.DataBaseConn.database_path)
            self.cursor = self.conn.cursor()
            logger.info("Successfully connected to the SQLite database.")
        except:
            raise Exception
        try:
            logger.info(f"Reading {self.DataBaseConn.table} table ")
            df = pd.read_sql_query(sql=self.DataBaseConn.query, con=self.conn)
            logger.info(f"Successfully read the {self.DataBaseConn.table} as pandas dataframe")
            return df
        except:
            raise Exception
        

@dataclass
class DataIngestionConfig:
    raw_data_path  = os.path.join('artifacts', 'raw_data.csv')
    train_data_path = os.path.join('artifacts', 'train_data.csv')
    test_data_path = os.path.join('artifacts', 'test_data.csv')


class DataIngestion:

    def __init__(self):
        self.data_config = DataIngestionConfig()
        pass
    
    def initate_data_ingestion(self):
        logger.info("Initiating data ingestion")

        try:
            db_handler = DataBaseHandler()
            raw_data = db_handler.DataFrameExraction()
            os.makedirs("artifacts")
            raw_data.to_csv(self.data_config.raw_data_path, header=True, index=False)
            logger.info(f"succesfully ingested the raw data as a csv file into {self.data_config.raw_data_path}")
            logger.info("Initiating train test split")
            train_data, test_data = train_test_split(raw_data, test_size=0.3, random_state=42)
            train_data.to_csv(self.data_config.train_data_path, header=True, index=False)
            test_data.to_csv(self.data_config.test_data_path, header=True, index=False)      
            logger.info(f"train and test data split successful and stored respectively as csv files at {self.data_config.train_data_path}, {self.data_config.test_data_path}")     
            return(
                self.data_config.raw_data_path,
                self.data_config.train_data_path,
                self.data_config.test_data_path
            )



        except:
            raise Exception