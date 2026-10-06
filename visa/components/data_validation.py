import json
import sys
import pandas as pd
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataQualityProfileSection
from pandas import DataFrame
from visa.exception import USvisaException
from visa.logger import logging
from visa.utils.main_utils import write_yaml_file,read_yaml_file
from visa.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from visa.entity.config_entity  import DataValidationConfig
from visa.constants import SCHEMA_FILE_PATH

class DataValidaton:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self._schema_config=read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e,sys)
    
    def validate_number_of_columns(self,dataframe:DataFrame)->bool:
        try:
            status=len(dataframe.columns)==len(self._schema_config['columns'])
            logging.info(f"Is Required column present : [{status}]")
            return status
        except Exception as e:
            raise USvisaException(e,sys)
    
    def is_column_exist(self,df:DataFrame)->bool:
        try:
            dataframe_columns=df.columns
            missing_numerical_columns=[]
            missing_categorical_columns=[]
            for column in self._schema_config['numerical_columns']:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)
            if len(missing_numerical_columns)>0:
                logging.info(f"Missing numerical column : {missing_numerical_columns}")
            
            for column in self._schema_config['categorical_columns']:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)
            if len(missing_categorical_columns)>0:
                logging.info(f"Missing categorical column : {missing_categorical_columns}")
            return False if len(missing_categorical_columns)>0 or len(missing_numerical_columns)>0 else True
        except Exception as e:
            raise USvisaException(e,sys) from e
    
    @staticmethod
    def read_data(file_path)->DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e,sys)
    