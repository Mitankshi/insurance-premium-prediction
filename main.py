from insurance.logger import logging
from insurance.exception import InsuranceException
import os,sys
from insurance.utils import get_collection_As_dataframe

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
            get_collection_As_dataframe(database_name = "INSURANCE", collection_name = "INSURANCE_PROJECT")

        except Exception as e:
            print(e)
