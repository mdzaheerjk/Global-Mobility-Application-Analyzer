from visa.entity.config_entity import ModelEvaluationConfig
from visa.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import f1_score
from visa.exception import USvisaException
from visa.logger import logging
from visa.constants import TARGET_COLUMN, CURRENT_YEAR, SCHEMA_FILE_PATH
import sys
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from visa.entity.estimator import TargetValueMapping, VisaModel
from visa.entity.s3_estimator import VisaEstimator
from visa.utils.main_utils import read_yaml_file, drop_columns

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: Optional[float]
    is_model_accepted: bool
    difference: float
    
class ModelEvaluation:
    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys) from e
    
    def get_base_model(self) -> Optional[VisaEstimator]:
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path
            usvisa_estimator = VisaEstimator(bucket_name=bucket_name, model_path=model_path)
            if usvisa_estimator.is_model_present(model_path=model_path):
                return usvisa_estimator
            return None
        except Exception as e:
            logging.warning(f"Could not check base model in S3: {e}")
            return None
    
    def evaluate_model(self) -> EvaluateModelResponse:
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            test_df['company_age'] = CURRENT_YEAR - test_df['yr_of_estab']
            
            drop_cols = self._schema_config.get('drop_columns', [])
            test_df = drop_columns(df=test_df, cols=drop_cols)
            
            x = test_df.drop(columns=[TARGET_COLUMN])
            y = test_df[TARGET_COLUMN].replace(TargetValueMapping()._asdict()).astype(int)
            
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            
            best_model_f1_score = None
            best_model = self.get_base_model()
            if best_model is not None:
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)
                
            tmp_best_model_score = 0.0 if best_model_f1_score is None else best_model_f1_score
            is_accepted = (trained_model_f1_score - tmp_best_model_score) >= self.model_eval_config.changed_threshold_score or best_model_f1_score is None
            
            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=is_accepted,
                difference=trained_model_f1_score - tmp_best_model_score
            )
            logging.info(f"Model evaluation result: {result}")
            return result
        except Exception as e:
            raise USvisaException(e, sys)
    
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path
            
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference
            )           
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e