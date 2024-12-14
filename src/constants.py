import pandas as pd
from src.components.data_ingestion import DataIngestionConfig

df_train = pd.read_csv(DataIngestionConfig.train_data_path)
df_train.dropna(inplace=True)

num_columns  = df_train.select_dtypes(include="Int64").columns
text_columns = df_train.select_dtypes(include="object").columns