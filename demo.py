from visa.pipeline.training_pipeline import TrainingPipeline
from visa.logger import logging

if __name__ == '__main__':
    try:
        logging.info("Starting demo training pipeline execution...")
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
        logging.info("Demo training pipeline executed successfully.")
        print("Training pipeline finished successfully.")
    except Exception as e:
        logging.error(f"Demo failed: {e}")
        print(f"Error: {e}")