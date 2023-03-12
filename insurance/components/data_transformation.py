from insurance.entity import artifact_entity, config_entity
from insurance.exception import InsuranceException
from insurance.logger import logging
from typing import Optional
import os
import sys
from sklearn.pipeline import Pipeline
import pandas as pd
from insurance import utils
import numpy as np
from sklearn.preprocessing import LabelEncoder
#from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from insurance.config import TARGET_COLUMN

# Missing values imputation
# Outliers Handling
# Imbalanced data handling
# Convert Categorical data into numerical data


class DataTransformation:

    def __init__(self, data_transformation_config: config_entity.DataTransformationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        logging.info(f"datatransformation class created")
        try:
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            logging.info(f"data ingestion artifact data created")
        except Exception as e:
            raise InsuranceException(e, sys)

    @classmethod
    def get_data_transformer_object(cls) -> Pipeline:  # Create cls class
        logging.info(f"get data transformer object created")
        try:
            logging.info(f"handling outliers")
            simple_imputer = SimpleImputer(strategy='constant', fill_value=0)
            robust_scaler = RobustScaler()
            pipeline = Pipeline(steps=[
                ('Imputer', simple_imputer),
                ('RobustScaler', robust_scaler)
            ])
            logging.info(f"outliers and missing values handled")
            return pipeline

        except Exception as e:
            raise InsuranceException(e, sys)

    def initiate_data_transformation(self) -> artifact_entity.DataTransformationArtifact:
        logging.info(f"inititate data transformation created")
        try:
            logging.info(f"reading train data")
            train_df = pd.read_csv(
                self.data_ingestion_artifact.train_file_path)
            logging.info(f"reading test data")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info(f"droping the target column from train data")
            input_feature_train_df = train_df.drop(TARGET_COLUMN, axis=1)
            logging.info(f"droping the target column from test data")
            input_feature_test_df = test_df.drop(TARGET_COLUMN, axis=1)

            logging.info(f"target column in train data")
            target_feature_train_df = train_df[TARGET_COLUMN]
            logging.info(f"target column in test data")
            target_feature_test_df = test_df[TARGET_COLUMN]

            label_encoder = LabelEncoder()

            logging.info(f"target feature in train data")
            target_feature_train_arr = target_feature_train_df.squeeze()
            logging.info(f"target feature in test data")
            target_feature_test_arr = target_feature_test_df.squeeze()

            logging.info(f"for loop for input feature train data")
            for col in input_feature_train_df.columns:
                if input_feature_test_df[col].dtypes == 'O':
                    input_feature_train_df[col] = label_encoder.fit_transform(
                        input_feature_train_df[col])
                    input_feature_test_df[col] = label_encoder.fit_transform(
                        input_feature_test_df[col])
                else:
                    input_feature_train_df[col] = input_feature_train_df[col]
                    input_feature_test_df[col] = input_feature_test_df[col]

            logging.info(f"creating transformation pipeline")
            transformation_pipleine = DataTransformation.get_data_transformer_object()
            logging.info(f"fit in the data in transform input train data")
            transformation_pipleine.fit(input_feature_train_df)

            logging.info(f"input feature train array")
            input_feature_train_arr = transformation_pipleine.transform(
                input_feature_train_df)
            logging.info(f"input feature test array")
            input_feature_test_arr = transformation_pipleine.transform(
                input_feature_test_df)

            logging.info(f"train array called")
            train_arr = np.c_[input_feature_train_arr,
                              target_feature_train_arr]
            logging.info(f"test array called")
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            logging.info(f"saving train data in array")
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transform_train_path,
                                        array=train_arr)
            logging.info(f"saving test data in array")
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transform_test_path,
                                        array=test_arr)

            logging.info(f"saving train data object")
            utils.save_object(file_path=self.data_transformation_config.transform_object_path,
                              obj=transformation_pipleine)
            logging.info(f"saving test data object")
            utils.save_object(file_path=self.data_transformation_config.target_encoder_object_path,
                              obj=label_encoder)

            logging.info(f"creating data transformation artifact")
            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                transformed_train_path=self.data_transformation_config.transform_train_path,
                transformed_test_path=self.data_transformation_config.transform_test_path,
                target_encoder_path=self.data_transformation_config.target_encoder_object_path

            )
            logging.info(f"completed data transformation")
            return data_transformation_artifact
        except Exception as e:
            raise InsuranceException(e, sys)
