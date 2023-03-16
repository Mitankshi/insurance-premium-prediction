import streamlit as st

import numpy as np
import pandas as pd

import pickle


model = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open('target_encoder.pkl', 'rb'))
transformer = pickle.load(open('transformer.pkl', 'rb'))

st.title("Insurance Premium Prediction")

Gender = st.selectbox("Please select your gender", ('male', 'female'))
