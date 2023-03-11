from insurance.entity import artifact_entity,config_entity
from insurance.exception import InsuranceException
from insurance.logger import logging
from typing import Optional
import os,sys 
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

    logging.info(f"we are creating data transformation class")
    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,
                    data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            self.data_transformation_config=data_transformation_config
            self.data_ingestion_artifact=data_ingestion_artifact
            logging.info(f"now we get into class method")
        except Exception as e:
            raise InsuranceException(e, sys)


    @classmethod
    def get_data_transformer_object(cls)->Pipeline: # Create cls class
        logging.info(f"we are creating class")
        try:
            simple_imputer = SimpleImputer(strategy='constant', fill_value=0)
            robust_scaler =  RobustScaler()
            pipeline = Pipeline(steps=[
                    ('Imputer',simple_imputer),
                    ('RobustScaler',robust_scaler)
                ])
            return pipeline
            logging.info(f"we have return the pipeline")
        except Exception as e:
            raise InsuranceException(e, sys)


    def initiate_data_transformation(self)->artifact_entity.DataTransformationArtifact:
        try:
            logging.info(f"here we do the transformation of train and test data")
            train_df=pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df=pd.read_csv(self.data_ingestion_artifact.test_file_path)
            
            logging.info(f"now we drop the target column")
            input_feature_train_df=train_df.drop(TARGET_COLUMN,axis=1)
            input_feature_test_df=test_df.drop(TARGET_COLUMN,axis=1)

            logging.info(f"now we feature the target column")
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            label_encoder = LabelEncoder()

            logging.info(f"target feature into array")
            target_feature_train_arr = target_feature_train_df.squeeze()
            target_feature_test_arr = target_feature_test_df.squeeze()

            logging.info(f"we run for loop for input feature")
            for col in input_feature_train_df.columns:
                if input_feature_test_df[col].dtypes == 'O':
                    input_feature_train_df[col] = label_encoder.fit_transform(input_feature_train_df[col])
                    input_feature_test_df[col] = label_encoder.fit_transform(input_feature_test_df[col])
                else:
                    input_feature_train_df[col] = input_feature_train_df[col]
                    input_feature_test_df[col] = input_feature_test_df[col]

            logging.info(f"we fit the data into transform")
            transformation_pipleine = DataTransformation.get_data_transformer_object()
            transformation_pipleine.fit(input_feature_train_df)

            logging.info(f"we made input feature into array")
            input_feature_train_arr = transformation_pipleine.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipleine.transform(input_feature_test_df)
            
            logging.info(f"we train our data to numpy")
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr ]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            logging.info(f"we save the transformation data into array")
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path,
                                        array=train_arr)

            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path,
                                        array=test_arr)

            logging.info(f"we save the pipeline into label encoder")
            utils.save_object(file_path=self.data_transformation_config.transform_object_path,
             obj=transformation_pipleine)

            utils.save_object(file_path=self.data_transformation_config.target_encoder_path,
            obj=label_encoder)


            logging.info(f"we are giving the artifact file location")
            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                transformed_train_path = self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path,
                target_encoder_path = self.data_transformation_config.target_encoder_path

            )

            return data_transformation_artifact
        
            logging.info(f"we are succesfully loaded the data")

        except Exception as e:
            raise InsuranceException(e, sys)