import json
import sys
import pandas as pd
from pandas import DataFrame
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric, DataDriftTable

from visa.exception import USvisaException
from visa.logger import logging
from visa.utils.main_utils import write_yaml_file, read_yaml_file, drop_columns
from visa.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from visa.entity.config_entity import DataValidationConfig
from visa.constants import SCHEMA_FILE_PATH

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys)
    
    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        try:
            status = len(dataframe.columns) == len(self._schema_config['columns'])
            logging.info(f"Is required number of columns present: [{status}]")
            return status
        except Exception as e:
            raise USvisaException(e, sys)
    
    def is_column_exist(self, df: DataFrame) -> bool:
        try:
            dataframe_columns = list(df.columns)
            missing_numerical_columns = []
            missing_categorical_columns = []
            
            for column in self._schema_config['numerical_columns']:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)
            if len(missing_numerical_columns) > 0:
                logging.info(f"Missing numerical columns: {missing_numerical_columns}")
            
            for column in self._schema_config['categorical_columns']:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)
            if len(missing_categorical_columns) > 0:
                logging.info(f"Missing categorical columns: {missing_categorical_columns}")
                
            return len(missing_categorical_columns) == 0 and len(missing_numerical_columns) == 0
        except Exception as e:
            raise USvisaException(e, sys) from e
    
    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)
    
    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame) -> bool:
        try:
            # Exclude ID and drop columns with unique identifier cardinality to avoid massive contingency matrix
            drop_cols = self._schema_config.get('drop_columns', [])
            cols_to_drop = [col for col in drop_cols if col in reference_df.columns]
            
            ref_data = drop_columns(df=reference_df.copy(), cols=cols_to_drop) if cols_to_drop else reference_df
            curr_data = drop_columns(df=current_df.copy(), cols=cols_to_drop) if cols_to_drop else current_df
            
            data_drift_report = Report(metrics=[DatasetDriftMetric(), DataDriftTable()])
            data_drift_report.run(reference_data=ref_data, current_data=curr_data)
            
            json_report = data_drift_report.as_dict()
            
            drift_metric = next((m for m in json_report.get('metrics', []) if m.get('metric') == 'DatasetDriftMetric'), None)
            table_metric = next((m for m in json_report.get('metrics', []) if m.get('metric') == 'DataDriftTable'), None)
            
            if drift_metric:
                n_features = drift_metric['result'].get('number_of_columns', len(ref_data.columns))
                n_drifted_features = drift_metric['result'].get('number_of_drifted_columns', 0)
                drift_status = drift_metric['result'].get('dataset_drift', False)
                share_drifted = drift_metric['result'].get('share_of_drifted_columns', 0.0)
            else:
                n_features = len(ref_data.columns)
                n_drifted_features = 0
                drift_status = False
                share_drifted = 0.0
                
            drift_by_columns = {}
            if table_metric:
                for col_name, col_data in table_metric.get('result', {}).get('drift_by_columns', {}).items():
                    drift_by_columns[col_name] = {
                        "drift_detected": col_data.get("drift_detected", False),
                        "drift_score": float(col_data.get("drift_score", 0.0))
                    }
                    
            summary_report = {
                "dataset_drift": bool(drift_status),
                "number_of_features": int(n_features),
                "number_of_drifted_features": int(n_drifted_features),
                "share_of_drifted_columns": float(share_drifted),
                "drift_by_columns": drift_by_columns
            }
            
            write_yaml_file(file_path=self.data_validation_config.drift_report_file_path, content=summary_report)
            logging.info(f"{n_drifted_features}/{n_features} drift detected (status: {drift_status})")
            return drift_status
        except Exception as e:
            raise USvisaException(e, sys) from e
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            validation_error_msg = ""
            logging.info("Starting data validation")
            train_df = DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            
            status = self.validate_number_of_columns(dataframe=train_df)
            logging.info(f"All required columns count present in training dataframe: {status}")
            if not status:
                validation_error_msg += "Columns count mismatch in training dataframe. "
                
            status = self.validate_number_of_columns(dataframe=test_df)
            logging.info(f"All required columns count present in testing dataframe: {status}")
            if not status:
                validation_error_msg += "Columns count mismatch in testing dataframe. "
            
            status = self.is_column_exist(df=train_df)
            if not status:
                validation_error_msg += "Required columns missing in training dataframe. "
                
            status = self.is_column_exist(df=test_df)
            if not status:
                validation_error_msg += "Required columns missing in testing dataframe. "
            
            validation_status = len(validation_error_msg) == 0
            
            if validation_status:
                drift_status = self.detect_dataset_drift(train_df, test_df)
                if drift_status:
                    logging.info("Drift Detected.")
                    validation_error_msg = "Drift Detected"
                else:
                    logging.info("Drift not detected.")
                    validation_error_msg = "Drift not detected"
            else:
                logging.info(f"Validation error: {validation_error_msg}")
            
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
            logging.info(f"Data Validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        
        except Exception as e:
            raise USvisaException(e, sys) from e

# Alias for backwards compatibility
DataValidaton = DataValidation