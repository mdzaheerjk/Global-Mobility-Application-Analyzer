from visa.constants import DATABASE_NAME
from visa.exception import USvisaException
from visa.logger import logging
import pandas as pd
import sys
import os
from typing import Optional
import numpy as np
from visa.configuration.mongo_db_connection import MongoDBClient

class VisaData:
    def __init__(self):
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            logging.warning(f"Could not connect to MongoDB: {e}")
            self.mongo_client = None
    
    def export_collection_as_dataframe(self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        try:
            df = None
            if self.mongo_client is not None and self.mongo_client.client is not None:
                if database_name is None:
                    collection = self.mongo_client.database[collection_name]
                else:
                    collection = self.mongo_client.client[database_name][collection_name]
                
                records = list(collection.find())
                if len(records) > 0:
                    df = pd.DataFrame(records)
                    if "_id" in df.columns:
                        df = df.drop(columns=['_id'])
                    df.replace({"na": np.nan}, inplace=True)
            
            if df is None or len(df) == 0:
                local_csv_path = os.path.join("notebooks", "Visadataset.csv")
                if os.path.exists(local_csv_path):
                    logging.info(f"Loading data from fallback local dataset: {local_csv_path}")
                    df = pd.read_csv(local_csv_path)
                    df.replace({"na": np.nan}, inplace=True)
                else:
                    raise Exception(f"No records found in MongoDB collection '{collection_name}' and local dataset not found.")
            
            return df
        except Exception as e:
            raise USvisaException(e, sys)