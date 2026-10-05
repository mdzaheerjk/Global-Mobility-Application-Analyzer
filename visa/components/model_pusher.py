import sys
from visa.cloud_storage.aws_storage import SimpleStorageService
from visa.exception import USvisaException
from visa.logger import logging
from visa.entity.artifact_entity import ModelPusherArtifact,ModelEvaluationArtifact
from visa.entity.config_entity import ModelPusherConfig
from visa.entity.s3_estimator import VisaEstimator

class ModelPusher:
    def __init__(self,model_evaluation_artifact:ModelEvaluationArtifact,
                 model_pusher_config:ModelPusherConfig):
        self.s3=SimpleStorageService()
        self.model_evaluation_artifact=model_evaluation_artifact
        self.model_pusher_config=model_pusher_config
        self.usvisa_estimator=VisaEstimator(bucket_name=model_pusher_config.bucket_name,
                                            model_path=model_pusher_config.s3_model_key_path)
    
    def initiate_model_pusher(self)->ModelPusherArtifact:
        logging.info("Entered initiate model_pusher method of ModelTrainer class")
        
        try:
            logging.info("Uploading artifact folder to s3 bucket")
            
            self.usvisa_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)
            
            model_pusher_artifact=ModelPusherArtifact(bucket_name=self.model_pusher_config.bucket_name,
                                                      s3_model_path=self.model_pusher_config.s3_model_key_path)
            logging.info("Uploadung artifact folder to s3 bucket")
            logging.info(f"Model Pusher artifact : [{model_pusher_artifact}]")
            logging.info("Excited initiate_model_pusher method of ModelTrainer class")
            
            return model_pusher_artifact
        except Exception as e:
            raise USvisaException(e,sys) from e
        