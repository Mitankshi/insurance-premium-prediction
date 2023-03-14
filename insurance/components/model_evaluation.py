from insurance.entity import artifact_entity, config_entity
from insurance.exception import InsuranceException
from insurance.logger import logging
from typing import Optional
import os
import sys
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from insurance import utils
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from insurance.predictor import ModelResolver


class ModelEvaluation:
    def __init__(self,
                 model_evaluation_config: config_entity.ModelEvaluationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact,
                 data_transformation_artifact: artifact_entity.DataTransformationArtifact,
                 model_trainer_artifact: artifact_entity.ModelTrainerArtifact):

        try:
            logging.info(f"calling the class ")
            self.model_evaluation_config = model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver()

        except Exception as e:
            raise InsuranceException(e, sys)

    def initiate_model_evaluation(self) -> artifact_entity.ModelEvaluationArtifact:
        try:
            logging.info(f"getting latest directory")
            latest_dir_path = self.model_resolver.get_latest_dir_path()

            logging.info(f"writting function")
            if latest_dir_path == None:
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(
                    is_model_accepted=True, improved_accuracy=None)
                logging.info(
                    f"model evaluation artifact: {model_eval_artifact}")
                logging.info(f"completed evaluation")
                return model_eval_artifact

        except Exception as e:
            raise InsuranceException(e, sys)
