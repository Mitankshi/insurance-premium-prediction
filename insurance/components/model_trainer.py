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

# model define & trainer
# accuracy
#overfitting and underfitting


class ModelTrainer:
    def __init__(self, model_trainer_config: config_entity.ModelTrainerConfig,
                 data_transformation_artifact: artifact_entity.DataTransformationArtifact):
        try:
            logging.info(f"creating model training file")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise InsuranceException(e, sys)

    # linear regression
    def train_model(self, X, y):
        try:
            logging.info(f"linear regression algorithm")
            lr = LinearRegression()
            lr.fit(X, y)
            logging.info(f"completed algorithm")
            return lr

        except Exception as e:
            raise InsuranceException(e, sys)

    def initiate_model_trainer(Self) -> artifact_entity.ModelTrainerArtifact:
        try:
            logging.info(f"we are taking transformed train data")
            train_arr = utils.load_numpy_array_data(
                file_path=Self.data_transformation_artifact.transformed_train_path)
            logging.info(f"we are taking transformed test data")
            test_arr = utils.load_numpy_array_data(
                file_path=Self.data_transformation_artifact.transformed_test_path)

            logging.info(f"we are splitting data in x train and y train data")
            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            logging.info(f"we are splitting data in x test and y test data")
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            logging.info(f"building the model")
            model = Self.train_model(X=x_train, y=y_train)

            logging.info(f"predicting the model for train data")
            yhat_train = model.predict(x_train)
            logging.info(f"checking r2 square for train data")
            r2_train_score = r2_score(y_true=y_train, y_pred=yhat_train)

            logging.info(f"predicting the model for test data")
            yhat_test = model.predict(x_test)
            logging.info(f"checking r2 square for test data")
            r2_test_score = r2_score(y_true=y_test, y_pred=yhat_test)

            logging.info(f"checking threshold")
            if r2_test_score < Self.model_trainer_config.expected_accuracy:
                raise Exception(
                    f"model is not good as it is not able to give expected accuracy:{Self.model_trainer_config.expected_accuracy}: model actual score: {r2_test_score}")

            logging.info(f"checking overfitting")
            diff = abs(r2_train_score - r2_test_score)

            if diff > Self.model_trainer_config.overfitting_threshold:
                raise Exception(
                    f"train model and test score diff: {diff} is more than overfitting threshold {Self.model_trainer_config.overfitting_threshold}")

            logging.info(f"saving the model")
            utils.save_object(
                file_path=Self.model_trainer_config.model_path, obj=model)

            logging.info(f"preparing the artifact for saving the model")
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(
                model_path=Self.model_trainer_config.model_path, r2_train_score=r2_train_score, r2_test_score=r2_test_score)

            logging.info(f"model training completed")
            return model_trainer_artifact

        except Exception as e:
            raise InsuranceException(e, sys)
