from visa.cloud_storage.aws_storage import SimpleStorageService
from visa.exception import USvisaException
from visa.entity.estimator import VisaModel
from visa.utils.main_utils import load_object
from visa.logger import logging
import sys
import os
from pandas import DataFrame

class VisaEstimator:
    def __init__(self, bucket_name, model_path):
        self.bucket_name = bucket_name
        self.model_path = model_path
        self.loaded_model: VisaModel = None
        self.s3 = None
        try:
            self.s3 = SimpleStorageService()
        except Exception as e:
            logging.warning(f"SimpleStorageService could not be initialized: {e}")
    
    def is_model_present(self, model_path=None) -> bool:
        try:
            target_path = model_path if model_path is not None else self.model_path
            if self.s3 is not None:
                return self.s3.s3_key_path_available(bucket_name=self.bucket_name, s3_key=target_path)
            return os.path.exists(target_path) or os.path.exists("model.pkl")
        except Exception as e:
            logging.warning(f"Error checking model presence: {e}")
            return os.path.exists(self.model_path) or os.path.exists("model.pkl")
    
    def load_model(self) -> VisaModel:
        if self.s3 is not None:
            try:
                return self.s3.load_model(self.model_path, bucket_name=self.bucket_name)
            except Exception as e:
                logging.warning(f"Failed to load model from S3, falling back to local: {e}")
        
        local_candidates = [
            self.model_path,
            "model.pkl",
            os.path.join("artifact", "model.pkl")
        ]
        if os.path.exists("artifact"):
            for root, dirs, files in os.walk("artifact"):
                if "model.pkl" in files:
                    local_candidates.append(os.path.join(root, "model.pkl"))
        
        for candidate in local_candidates:
            if os.path.exists(candidate):
                logging.info(f"Loading local model from {candidate}")
                return load_object(candidate)
        
        raise Exception(f"Model file '{self.model_path}' not found in S3 or local directory.")
    
    def save_model(self, from_file, remove: bool = False) -> None:
        try:
            if self.s3 is not None:
                self.s3.upload_file(
                    from_file,
                    to_filename=self.model_path,
                    bucket_name=self.bucket_name,
                    remove=remove
                )
            else:
                logging.warning("S3 client not initialized. Model not uploaded to S3.")
        except Exception as e:
            raise USvisaException(e, sys)
    
    def predict(self, dataframe: DataFrame):
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe=dataframe)
        except Exception as e:
            raise USvisaException(e, sys)