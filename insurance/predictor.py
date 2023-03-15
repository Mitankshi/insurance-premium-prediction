import os
from insurance.logger import logging
from typing import Optional
from insurance.entity.config_entity import TARGET_ENCODER_OBJECT_FILE_NAME, TRANSFORMER_OBJECT_FILE_NAME, MODEL_FILE_NAME


# folder (new model save)
# comparision new model vs oldmodel
# accepted else rejected


class ModelResolver:
    def __init__(self, model_registry: str = "saved_models",
                 transformer_dir_name="transformer",
                 target_encoder_dir_name="target_encoder",
                 model_dir_name="model"):
        logging.info(f"we are making new folder for new data")
        self.model_registry = model_registry
        os.makedirs(self.model_registry, exist_ok=True)
        self.transformer_dir_name = transformer_dir_name
        self.target_encoder_dir_name = target_encoder_dir_name
        self.model_dir_name = model_dir_name

    def get_latest_dir_path(self) -> Optional[str]:
        try:
            logging.info(f"calling the saved directory")
            dir_name = os.listdir(self.model_registry)
            logging.info(f"writting function")
            if len(dir_name) == 0:
                return None

            logging.info(f"mapping the directory")
            dir_name = list(map(int, dir_name))
            logging.info(f"calling latest directory name")
            latest_dir_name = max(dir_name)

            return os.path.join(self.model_registry, f"{latest_dir_name}")

        except Exception as e:
            raise e

    def get_latest_model_path(self):
        try:
            logging.info(f"defining get latest directory")
            latest_dir = self.get_latest_dir_path()
            logging.info(f"writing function for model path")
            if latest_dir is None:
                raise Exception(f"model is not available")
            return os.path.join(latest_dir, self.model_dir_name, MODEL_FILE_NAME)

        except Exception as e:
            raise e

    def get_latest_transformer_path(Self):
        try:
            logging.info(f"defining get latest transformer")
            latest_dir = Self.get_latest_dir_path()
            logging.info(f"writing function for transformer path")
            if latest_dir is None:
                raise Exception(f"function transform data is not available")
            return os.path.join(latest_dir, Self.transformer_dir_name, TRANSFORMER_OBJECT_FILE_NAME)

        except Exception as e:
            raise e

    def get_latest_target_encoder_path(self):
        try:
            logging.info(f"defining get latest encoder")
            latest_dir = self.get_latest_dir_path()
            logging.info(f"writing function for encoder path")
            if latest_dir is None:
                raise Exception(f"target encoder data is not available")
            return os.path.join(latest_dir, self.target_encoder_dir_name, TARGET_ENCODER_OBJECT_FILE_NAME)

        except Exception as e:
            raise e

    def get_latest_save_dir_path(self) -> str:
        try:
            logging.info(f"latest directory saving")
            latest_dir = self.get_latest_dir_path()
            logging.info(f"writing function for saving path")
            if latest_dir == None:
                logging.info(f"inside if condition")
                return os.path.join(self.model_registry, f"{0}")

            logging.info(f"checking latest directory")
            latest_dir_num = int(os.path.basename(self.get_latest_dir_path()))
            return os.path.join(self.model_registry, f"{latest_dir_num+1}")

        except Exception as e:
            raise e

    def get_latest_save_model_path(Self):
        try:
            logging.info(f"saving model data")
            latest_dir = Self.get_latest_save_dir_path()
            # pkl
            return os.path.join(latest_dir, Self.model_dir_name, MODEL_FILE_NAME)
        except Exception as e:
            raise e

    def get_latest_save_transform_path(self):
        try:
            logging.info(f"saving transform data")
            latest_dir = self.get_latest_save_dir_path()
            # pkl
            return os.path.join(latest_dir, self.transformer_dir_name, TRANSFORMER_OBJECT_FILE_NAME)
        except Exception as e:
            raise e

    def get_latest_save_encoder_path(Self):
        try:
            logging.info(f"saving encoder data")
            latest_dir = Self.get_latest_save_dir_path()
            # pkl

            logging.info(f"after return")
            return os.path.join(latest_dir, Self.target_encoder_dir_name, TARGET_ENCODER_OBJECT_FILE_NAME)
        except Exception as e:
            raise e
