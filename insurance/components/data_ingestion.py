import pandas as pd
import numpy as np
import os
import sys
from insurance.entity import config_entity
from insurance.entity import artifact_entity
from insurance.exception import InsuranceException
from insurance import utils
from insurance.logger import logging
from sklearn.model_selection import train_test_split


class DataIngestion:  # data division into train , test and validate
    def __init__(self, data_ingestion_config: config_entity.DataIngestionconfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise InsuranceException(e, sys)

    def initiate_data_ingestion(self) -> artifact_entity.DataIngestionArtifact:
        try:
            logging.info(f"export collection data as pandas dataframe")
            df: pd.DataFrame = utils.get_collection_As_dataframe(
                database_name=self.data_ingestion_config.database_name,
                collection_name=self.data_ingestion_config.collection_name)
            logging.info(f"save data in future store")

            # replace na with NAN
            df.replace(to_replace="na", value=np.NAN, inplace=True)

            # save data in feature store
            logging.info(f"create feature store folder if not available")
            feature_store_dir = os.path.dirname(
                self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok=True)

            # save df to feature store folder
            logging.info(f"save df to feature store folder")
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_file_path,
                      index=False, header=True)

            logging.info(f"splitting our dataset into train and test set")
            train_df, test_df = train_test_split(
                df, test_size=self.data_ingestion_config.test_size, random_state=1)

            logging.info(f"create dataset directory folder if not exists")
            dataset_dir = os.path.dirname(
                self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir, exist_ok=True)

            logging.info(f"save dataset to feature store folder")
            train_df.to_csv(
                path_or_buf=self.data_ingestion_config.train_file_path, index=False, header=True)
            test_df.to_csv(
                path_or_buf=self.data_ingestion_config.test_file_path, index=False, header=True)

            # prepare artifact folder
            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise InsuranceException(error_message=e, error_detail=sys)
