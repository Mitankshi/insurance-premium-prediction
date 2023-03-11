import os , sys
from insurance.exception import InsuranceException
from insurance.logger import logging
from datetime import datetime

FILE_NAME = "insurance.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TRANSFORMER_OBJECT_FILE_NAME = "transformer.pkl"
TARGET_ENCODER_OBJECT_FILE_NAME= "target_encoder.pkl"


class TrainingPipelineConfig:
    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
        except Exception as e:
            raise InsuranceException(e,sys)

class DataIngestionconfig:
    def __init__(self, training_pipeline_config : TrainingPipelineConfig):
        try:
            self.database_name = "INSURANCE"
            self.collection_name = "INSURANCE_PROJECT"
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, "data_ingestion")
            self.feature_store_file_path = os.path.join(self.data_ingestion_dir, "feature_store", FILE_NAME)
            self.train_file_path = os.path.join(self.data_ingestion_dir, "dataset", TRAIN_FILE_NAME)
            self.test_file_path = os.path.join(self.data_ingestion_dir, "dataset", TEST_FILE_NAME)
            self.test_size = 0.2
        except Exception as e:
            raise InsuranceException(e,sys)

# convert data into dictionary 
    def to_Dict(self)->dict:
        try:
            return self.__dict__
        except Exception as e:
            raise InsuranceException(e,sys)


class DataValidationConfig:

    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir, "data_validation")
        self.report_file_path = os.path.join(self.data_validation_dir, "report.yaml") #yaml,json,csv
        self.base_file_path = os.path.join("insurance.csv")
        self.missing_threshold:float = 0.2


class DataTransformationConfig:

    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir, "data_transformation")
        self.transform_object_path = os.path.join(self.data_transformation_dir, "transformer", TRANSFORMER_OBJECT_FILE_NAME)
        self.transform_train_path = os.path.join(self.data_transformation_dir, "transformed", TRAIN_FILE_NAME.replace("csv", "npz"))
        self.transform_test_path = os.path.join(self.data_transformation_dir, "transformed", TEST_FILE_NAME.replace("csv", "npz"))
        self.target_encoder_object_path = os.path.join(self.data_transformation_dir, "traget_encoder", TARGET_ENCODER_OBJECT_FILE_NAME)