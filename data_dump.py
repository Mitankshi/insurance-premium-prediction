import pymongo
import pandas as pd
import json


client = pymongo.MongoClient("mongodb+srv://mitankshi143:mita143@cluster0.9o9eeyi.mongodb.net/?retryWrites=true&w=majority")
db = client.test

DATA_FILE_PATH =(r"C:\Users\kundan Gupta\Downloads\ml_projects\insurance_premium_prediction\insurance.csv")
DATABASE_NAME = "INSURANCE"
COLLECTION_NAME = "INSURANCE_PROJECT"


if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and columns : {df.shape}")

    # Convert dataframe into json to dump in mongodb
    df.reset_index(drop=True, inplace=True)

    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])

    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)