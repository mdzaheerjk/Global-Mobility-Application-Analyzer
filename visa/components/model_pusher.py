import sys
from visa.cloud_storage.aws_storage import SimpleStorageService
from visa.exception import USvisaException
from visa.logger import logging
from visa.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from visa.entity.config_entity import ModelPusherConfig
from visa.entity.s3_estimator import VisaEstimator

class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        try:
            self.s3 = SimpleStorageService()
            self.usvisa_estimator = VisaEstimator(
                bucket_name=model_pusher_config.bucket_name,
                model_path=model_pusher_config.s3_model_key_path
            )
        except Exception as e:
            self.s3 = None
            self.usvisa_estimator = None
            logging.warning(f"S3 not available for ModelPusher: {e}")
    
    def initiate_model_pusher(self) -> ModelPusherArtifact:
        logging.info("Entered initiate_model_pusher method of ModelPusher class")
        
        try:
            if self.usvisa_estimator is not None:
                logging.info(f"Uploading model to S3 bucket: {self.model_pusher_config.bucket_name}")
                self.usvisa_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)
            else:
                logging.warning("S3 credentials not set; model is saved locally and will not be pushed to S3.")
            
            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.s3_model_key_path
            )
            logging.info(f"Model Pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelPusher class")
            
            return model_pusher_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e