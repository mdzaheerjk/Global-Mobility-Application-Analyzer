import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder,PowerTransformer
from sklearn.compose import ColumnTransformer
from visa.constants import TARGET_COLUMN,SCHEMA_FILE_PATH,CURRENT_YEAR
from visa.entity.config_entity import DataTransformationConfig
from visa.entity.artifact_entity import DataTransformationArtifact,DataIngestionArtifact,DataValidationArtifact
from visa.logger import logging
from visa.exception import USvisaException
from visa.utils.main_utils import save_object,save_numpy_array_data,read_yaml_file,drop_columns
from visa.entity.estimator import TargetValueMapping

class DataTransformation:
    def __init__(self,
                 data_ingestion_artifact:DataIngestionArtifact,
                 data_transformation_config:DataTransformationConfig,
                 data_validation_artifact:DataValidationArtifact):
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_transformation_config=data_transformation_config
            self.data_validation_artifact=data_validation_artifact
            self._schema_config=read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e,sys)
    
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e,sys)
    
    def get_data_transformer_object(self)->Pipeline:
        logging.info(
            "Entered get_data_transformer_object method of DataTransformation class"
        )
        try:
            logging.info("Got Numerical cols from schema config")
            
            numeric_transformer=StandardScaler()
            oh_transformer=OneHotEncoder()
            ordinal_encoder=OrdinalEncoder()
            
            logging.info("Initialized StandardScaler, OneHotEncoder, OrdinalEncoder")
            
            oh_columns=self._schema_config['oh_columns']
            or_columns=self._schema_config['or_columns']
            transform_columns=self._schema_config['num_features']
            
            logging.info("Initialize PowerTransformer")
            
            transform_pipe=Pipeline(steps=[
                ('transformer',PowerTransformer(method='yeo-johnson'))
            ])
            preprocessor=ColumnTransformer(
                [
                    ("OneHotEncoder",oh_transformer,oh_columns),
                    ("Ordinal_Encoder",ordinal_encoder,or_columns),
                    ("Transformer",transform_pipe,transform_columns)
                ]
            )
            logging.info("Created Preprocessor object from ColumnTransformer")
            
            logging.info(
                "Exited get_data_transformer_object method of DataTransformation class"
            )
            return preprocessor
        except Exception as e:
            raise USvisaException(e,sys) from e
        