import streamlit as st
# import preprocessor,helper

import numpy as np
import pandas as pd

import pickle
import xgboost as xg
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

model = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open('target_encoder.pkl', 'rb'))
transformer = pickle.load(open('transformer.pkl', 'rb'))

st.title("Insurance Premium Prediction")

Gender = st.selectbox("Please select your gender", ('male', 'female'))

age = st.text_input("Enter your age", 23)
age = int(age)

bmi = st.text_input("Enter your BMI", 20)
bmi = float(bmi)

children = st.selectbox(
    "Please select number of children", {0, 1, 2, 3, 4, 5})
children = int(children)

smoker = st.selectbox("Please select smoker category", {'yes', 'no'})

region = st.selectbox("Please select your region", {
                      "southwest", "southeast", "northwest", "northeast"})

l = {}
l['age'] = age
l['Gender'] = Gender
l['bmi'] = bmi
l['children'] = children
l['smoker'] = smoker
l['region'] = region

df = pd.DataFrame(l, index=[0])

df['region'] = encoder.transform(df['region'])
df['Gender'] = df['Gender'].map({'male': 1, 'female': 0})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

print(df)

df = transformer.transform(df)

y_pred = model.predict(df)

if st.button("Show Results"):
    st.header(f"round{y_pred[0], 2} INR")
