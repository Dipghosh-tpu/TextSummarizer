from src.textSummarizer.logging import logger
from src.textSummarizer.pipeline.stage_1_data_ingestion_pipeline import DataIngestionTrainingPipeline

STAGE_NAME = "data Ingestion Stage"

try:
    logger.info(f"stage {STAGE_NAME} initiated")

    data_ingestion_pipeline = DataIngestionTrainingPipeline()
    data_ingestion_pipeline.initiate_data_ingestion()
    logger.info(f"stage {STAGE_NAME} completed")
except Exception as e:
    logger.error(e)
    raise e