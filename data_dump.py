import pymongo
import pandas as pd
import json


client = pymongo.MongoClient("mongodb+srv://mitankshi143:mita143@cluster0.9o9eeyi.mongodb.net/?retryWrites=true&w=majority")
db = client.test

DATA_FILE_PATH =(r"C:\Users\kundan Gupta\OneDrive\Desktop\ml projects\insurance-premium-prediction\insurance.csv")
DATABASE_NAME = "INSURANCE"
COLLECTION_NAME = "INSURANCE_PROJECT"


if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and Columns: {df.shape}")

    df.rest_index(drop=True,inplace = True)

    json_record = list(json.load(df.T.to_json()).values())

    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)