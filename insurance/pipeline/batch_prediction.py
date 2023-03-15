from insurance.exception import InsuranceException
from insurance.logger import logging
import numpy as np
import pandas as pd
import os
import sys
from insurance.predictor import ModelResolver
from insurance.utils import load_object
from datetime import datetime

PREDICTION_DIR = "prediction"


def start_batch_prediction(input_file_path):
    try:
        logging.info(f"loading the path path")
        os.makedirs(PREDICTION_DIR, exist_ok=True)
        logging.info(f"collecting the file")
        model_resolver = ModelResolver(model_registry="saved_models")

        logging.info(f"reading the data")
        df = pd.read_csv(input_file_path)

        logging.info(f"handling null values")
        df.replace({"na": np.NAN}, inplace=True)

        logging.info(f"validating the data")
        # data_validation
        transformer = load_object(
            file_path=model_resolver.get_latest_transformer_path())

        logging.info(f"calling target encoder")
        target_encoder = load_object(
            file_path=model_resolver.get_latest_target_encoder_path())

        logging.info(f"encoding the object")
        input_feature_names = list(transformer.feature_names_in_)
        for i in input_feature_names:
            if df[i].dtypes == 'object':
                df[i] = target_encoder.fit_transform(df[i])

        logging.info(f"input array")
        input_arr = transformer.transform(df[input_feature_names])

        logging.info(f"for prediction data")
        model = load_object(file_path=model_resolver.get_latest_model_path())

        logging.info(f"calling prediction file")
        prediction = model.predict(input_arr)

        logging.info(f"defining prediction file")
        df['prediction'] = prediction

        logging.info(f"main function")
        prediction_file_name = os.path.basename(input_file_path).replace(
            ".csv", f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_path = os.path.join(
            PREDICTION_DIR, prediction_file_name)
        df.to_csv(prediction_file_path, index=False, header=True)

        return prediction_file_path

    except Exception as e:
        raise InsuranceException(e, sys)
