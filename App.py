# app.py: Streamlit application for salary prediction using the trained model

import streamlit as st
import joblib
import pandas as pd

# Load the pre-trained model from the saved file
try:
    model = joblib.load('best_model.pkl')
except FileNotFoundError:
    st.error("Model file 'best_model.pkl' not found. Please ensure it is in the same directory.")
    st.stop()

# Set up the Streamlit app layout
st.title('Salary Prediction App')
st.write('This app predicts salary based on experience, test score, and interview score using a trained linear regression model.')

# Input fields for user data
experience = st.slider('Years of Experience', min_value=0, max_value=15, value=0, step=1)
test_score = st.slider('Test Score (out of 10)', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
interview_score = st.slider('Interview Score (out of 10)', min_value=0.0, max_value=10.0, value=5.0, step=0.1)

# Button to trigger prediction
if st.button('Predict Salary'):
    # Prepare the input data as a DataFrame matching the model's expected features
    input_data = pd.DataFrame({
        'experience': [experience],
        'test_score(out of 10)': [test_score],
        'interview_score(out of 10)': [interview_score]
    })
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display the result
    st.success(f'Predicted Salary: ${prediction[0]:,.2f}')

# Footer information
st.write('---')
st.write('Model trained on hiring dataset. For demonstration purposes only.')
