from insurance.logger import logging
from insurance.exception import InsuranceException
from insurance.utils import get_collection_As_dataframe
import os
import sys
from insurance.entity import config_entity
from insurance.components.data_ingestion import DataIngestion
from insurance.components.data_validation import DataValidation
from insurance.components.data_transformation import DataTransformation
from insurance.components.model_trainer import ModelTrainer
from insurance.components.model_evaluation import ModelEvaluation
from insurance.components.model_pusher import ModelPusher


def start_training_pipeline():
    try:
        logging.info(f"calling all pipelines")
        training_pipeline_config = config_entity.TrainingPipelineConfig()

        logging.info(f"calling all the data ingestion pipelines")
        data_ingestion_config = config_entity.DataIngestionconfig(
            training_pipeline_config=training_pipeline_config)
        print(data_ingestion_config.to_Dict())
        data_ingestion = DataIngestion(
            data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        logging.info(f"calling all the data validation pipelines")
        data_validation_config = config_entity.DataValidationConfig(
            training_pipeline_config=training_pipeline_config)
        data_validation = DataValidation(data_validation_config=data_validation_config,
                                         data_ingestion_artifact=data_ingestion_artifact)
        data_validation_artifact = data_validation.initiate_data_validation()

        logging.info(f"calling all the data transformation pipelines")
        data_transformation_config = config_entity.DataTransformationConfig(
            training_pipeline_config=training_pipeline_config)
        data_transformation = DataTransformation(data_transformation_config=data_transformation_config,
                                                 data_ingestion_artifact=data_ingestion_artifact)
        data_transformation_artifact = data_transformation.initiate_data_transformation()

        logging.info(f"calling all the model trainer pipelines")
        model_trainer_config = config_entity.ModelTrainerConfig(
            training_pipeline_config=training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config,
                                     data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()

        logging.info(f"calling all the model evaluation pipelines")
        model_evaluation_config = config_entity.ModelEvaluationConfig(
            training_pipeline_config=training_pipeline_config)
        model_evaluation = ModelEvaluation(model_evaluation_config=model_evaluation_config,
                                           data_ingestion_artifact=data_ingestion_artifact,
                                           data_transformation_artifact=data_transformation_artifact,
                                           model_trainer_artifact=model_trainer_artifact)
        model_evaluation_artifact = model_evaluation.initiate_model_evaluation()

        logging.info(f"calling all the model pusher pipelines")
        model_pusher_config = config_entity.ModelPusherConfig(
            training_pipeline_config=training_pipeline_config)
        model_pusher = ModelPusher(model_pusher_config=model_pusher_config,
                                   data_transformation_artifact=data_transformation_artifact,
                                   model_trainer_artifact=model_trainer_artifact)
        model_pusher_artifact = model_pusher.inititate_model_pusher()

    except Exception as e:
        raise InsuranceException(e, sys)
