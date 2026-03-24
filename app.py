import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(5,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Load the trained federated model
@st.cache_resource
def load_model():
    model = build_model()
    model.load_weights("model_weights.weights.h5")
    return model

model = load_model()


st.title('Remote Patient Monitoring - Abnormal Heart Rate Detector')
st.write('Enter patient health data to predict the likelihood of an abnormal heart rate.')

# Define the features for the model
feature_cols = ['Heart Rate (BPM)', 'Blood Oxygen Level (%)', 'Step Count', 'Sleep Duration (hours)', 'Stress Level']

# --- Input fields for health metrics ---
st.header('Patient Health Metrics')

heart_rate = st.slider('Heart Rate (BPM)', min_value=30.0, max_value=200.0, value=70.0, step=0.1)
blood_oxygen = st.slider('Blood Oxygen Level (%)', min_value=85.0, max_value=100.0, value=97.0, step=0.1)
step_count = st.slider('Step Count', min_value=0.0, max_value=20000.0, value=5000.0, step=10.0)
sleep_duration = st.slider('Sleep Duration (hours)', min_value=2.0, max_value=12.0, value=7.0, step=0.1)
stress_level = st.slider('Stress Level (1-10)', min_value=1.0, max_value=10.0, value=5.0, step=1.0)

st.markdown("### Clinical Interpretation")
if heart_rate > 110:
    st.write("⚠️ High heart rate detected")
if blood_oxygen < 92:
    st.write("⚠️ Low blood oxygen detected")
if stress_level > 8:
    st.write("⚠️ High stress level detected")
    
# Create a DataFrame from the input values
input_data = pd.DataFrame([{
    'Heart Rate (BPM)': heart_rate,
    'Blood Oxygen Level (%)': blood_oxygen,
    'Step Count': step_count,
    'Sleep Duration (hours)': sleep_duration,
    'Stress Level': stress_level
}])

# --- Scaling the input data ---
@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

scaler = load_scaler()

# Transform the input data using the scaler
scaled_input_data = scaler.transform(input_data)

THRESHOLD = 0.6

# Make prediction
if st.button('Predict'):
    prediction = model.predict(scaled_input_data)
    prob = prediction[0][0]

    st.subheader('Prediction Result')
    st.write(f"Probability of Abnormal Heart Rate: {prob:.4f}")

    #final classification using trained threshold
    if prob >= THRESHOLD:
        st.error('🔴 Abnormal Heart Rate Detected')
    else:
        st.success('🟢 Normal Heart Rate')

    #risk interpretation
    st.markdown('### Risk level')
    if prob >= 0.75:
        st.error("High risk")
    elif prob >= 0.6:
        st.warning("Moderate risk")
    else:
        st.success("Low risk")

    st.write("\n--- Raw Input Data ---")
    st.dataframe(input_data)

    st.write("\n--- Scaled Input Data ---")
    st.dataframe(pd.DataFrame(scaled_input_data, columns=feature_cols))
