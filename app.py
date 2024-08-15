import streamlit as st
from joblib import load
import numpy as np
model = load('crop_recommedation_model (1).pkl')
st.title('Crop Forecasting Model App')
N = st.number_input('Nitrogen')
P = st.number_input('Phosphorus')
K = st.number_input('K')
temperature = st.number_input(' temperature')
humidity = st.number_input('humidity')
ph = st.number_input('ph')
rainfall = st.number_input('rainfall')
input_data = np.array([[N,P,K,temperature,humidity,ph,rainfall]])
prediction = model.predict(input_data)
Crop_types = ('rice' 'maize' 'chickpea' 'kidneybeans' 'pigeonpeas' 'mothbeans'
 'mungbean' 'blackgram' 'lentil' 'pomegranate' 'banana' 'mango' 'grapes'
 'watermelon' 'muskmelon' 'apple' 'orange' 'papaya' 'coconut' 'cotton'
 'jute' 'coffee')
st.write(f'Predicted Crop : {prediction[0]}')
