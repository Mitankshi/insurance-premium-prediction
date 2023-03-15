from insurance.entity import artifact_entity, config_entity
from insurance.exception import InsuranceException
from insurance.logger import logging
from typing import Optional
import os
import sys
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from insurance.predictor import ModelResolver
from insurance.utils import load_object
from insurance.config import TARGET_COLUMN


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

            # find location of previous model

            logging.info(f"finding previous model")
            transformer_path = self.model_resolver.get_latest_transformer_path()
            model_path = self.model_resolver.get_latest_model_path()
            target_encoder_path = self.model_resolver.get_latest_target_encoder_path()

            logging.info(f"calling load object")
            transformer = load_object(file_path=transformer_path)
            model = load_object(file_path=model_path)
            target_encoder = load_object(file_path=target_encoder_path)

            # current model

            logging.info(f"defining for new model")
            current_transformer = load_object(
                file_path=self.data_transformation_artifact.transform_object_path)
            current_model = load_object(
                file_path=self.model_trainer_artifact.model_path)
            current_target_encoder = load_object(
                file_path=self.data_transformation_artifact.target_encoder_path)

            logging.info(f"reading the data")
            test_Df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info(f"removing the atrget column as dependent column")
            target_df = test_Df[TARGET_COLUMN]
            y_true = target_df

            logging.info(f"defining feature")
            input_feature_name = list(transformer.feature_names_in_)

            logging.info(f"for loop running")
            for i in input_feature_name:
                if test_Df[i].dtypes == 'O':
                    test_Df[i] = target_encoder.fit_transform(test_Df[i])

            logging.info(f"transforming input data")
            input_arr = transformer.transform(test_Df[input_feature_name])

            logging.info(f"prediction of data")
            y_pred = model.predict(input_arr)

            # comparison b/w models
            logging.info(f"checking accuracy")
            previous_model_score = r2_score(y_true=y_true, y_pred=y_pred)

            # accuracy for current model
            logging.info(f"input data")
            input_feature_name = list(current_transformer.feature_names_in_)
            input_arr = current_transformer.transform(
                test_Df[input_feature_name])

            logging.info(f"prediction of current data")
            y_pred = current_model.predict(input_arr)
            y_true = target_df

            logging.info(f"checkign score for new model")
            current_model_score = r2_score(y_true=y_true, y_pred=y_pred)

            logging.info(f"comparision of models")
            if current_model_score <= previous_model_score:
                logging.info(
                    f"current trained model is not bettr than previous model")
                raise Exception(
                    f"current model is not better than previous model")

            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(
                is_model_accepted=True, improved_accuracy=current_model_score-previous_model_score)

            return model_eval_artifact

        except Exception as e:
            raise InsuranceException(e, sys)
