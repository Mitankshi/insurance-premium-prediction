# batch prediction
# training pipeline

from insurance.pipeline.batch_prediction import start_batch_prediction
from insurance.pipeline.training_pipeline import start_training_pipeline

#FILE_PATH = r"C:\Users\kundan Gupta\Downloads\ml_projects\insurance_premium_prediction\insurance.csv"

if __name__ == "__main__":
    try:
        #output = start_batch_prediction(input_file_path=FILE_PATH)
        output = start_training_pipeline()
        print(output)
    except Exception as e:
        print(e)
