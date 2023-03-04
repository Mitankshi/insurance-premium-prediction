import pandas as pd
import numpy as np
import os
import sys
import yaml
import dill
from insurance.exception import InsuranceException
from insurance.config import mongo_client
from insurance.logger import logging


def get_collection_As_dataframe(database_name:str, collection_name:str)->pd.DataFrame:
    try:
        logging.info(f"reading data from database:{database_name} and collection:{collection_name}")
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info(f"find columns:{df.columns}")
        if "_id" in df.columns:
            logging.info(f"dropping column: _id")
            df = df.drop("_id",axis=1)
        logging.info(f"Rows and Coilumns in df:{df.shape}")
        return df

    except Exception as e:
        raise InsuranceException(e,sys)





