from insurance.logger import logging
from insurance.exception import InsuranceException
import os,sys
from insurance.utils import get_collection_As_dataframe
from insurance.entity.config_entity import DataIngestionconfig
from insurance.entity import config_entity


#def test_logger_and_exception():
    #try:
        #logging.info("starting the test_logger_and_exception")
        #result = 3 / 0
        #print(result)
        #logging.info("ending point of the test_logger_and_exception")
    #except Exception as e:
        #logging.debug(str(e))
        #raise InsuranceException(e,sys)


if __name__ == "__main__":
        try:
           # get_collection_As_dataframe(database_name = "INSURANCE", collection_name = "INSURANCE_PROJECT")
           training_pipeline_config = config_entity.TrainingPipelineConfig()
           data_ingestion_config = config_entity.DataIngestionconfig(training_pipeline_config=training_pipeline_config)
           print(data_ingestion_config.to_Dict())

        except Exception as e:
            print(e)
