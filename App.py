# app.py: Enhanced Streamlit application for salary prediction with visualizations

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Salary Prediction Model", layout="centered")

# Load the pre-trained model
try:
    model = joblib.load('best_model.pkl')
except FileNotFoundError:
    st.error("Model file 'best_model.pkl' not found. Please ensure it is in the same directory.")
    st.stop()

# App title and description
st.title('Salary Prediction Model')
st.write("""
This application uses a linear regression model trained on hiring data to predict candidate salary 
based on years of experience, written test score, and interview performance score.
""")

# Sidebar for user inputs
st.sidebar.header('Input Candidate Features')

experience = st.sidebar.slider('Years of Experience', min_value=0, max_value=15, value=5, step=1)
test_score = st.sidebar.slider('Test Score (out of 10)', min_value=0.0, max_value=10.0, value=7.0, step=0.1)
interview_score = st.sidebar.slider('Interview Score (out of 10)', min_value=0.0, max_value=10.0, value=8.0, step=0.1)

# Display selected inputs
st.sidebar.write('### Selected Values')
st.sidebar.write(f"Experience: **{experience}** years")
st.sidebar.write(f"Test Score: **{test_score}/10**")
st.sidebar.write(f"Interview Score: **{interview_score}/10**")

# Prediction button
if st.button('Predict Salary', type="primary"):
    # Prepare input data
    input_data = pd.DataFrame({
        'experience': [experience],
        'test_score(out of 10)': [test_score],
        'interview_score(out of 10)': [interview_score]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Display prediction prominently
    st.success(f'Predicted Salary: **${prediction:,.2f}**')

    # Visualization 1: Feature contributions (model coefficients)
    st.markdown("### Model Insight: Feature Impact on Salary")
    coeffs = pd.DataFrame({
        'Feature': ['Experience (years)', 'Test Score', 'Interview Score'],
        'Coefficient': model.coef_
    })
    
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=coeffs, x='Coefficient', y='Feature', palette='viridis', ax=ax1)
    ax1.set_title('Impact of Each Feature on Predicted Salary')
    ax1.set_xlabel('Coefficient Value (Salary Increase per Unit)')
    st.pyplot(fig1)

    # Visualization 2: Effect of varying experience (fixing other scores)
    st.markdown("### Salary Trend by Experience")
    exp_range = np.arange(0, 16, 1)
    salary_by_exp = []
    
    for exp in exp_range:
        temp_input = pd.DataFrame({
            'experience': [exp],
            'test_score(out of 10)': [test_score],
            'interview_score(out of 10)': [interview_score]
        })
        salary_by_exp.append(model.predict(temp_input)[0])
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(exp_range, salary_by_exp, marker='o', linewidth=2, color='#2E86AB')
    ax2.axvline(x=experience, color='red', linestyle='--', label=f'Your Input ({experience} years)')
    ax2.set_title('Predicted Salary vs Years of Experience')
    ax2.set_xlabel('Years of Experience')
    ax2.set_ylabel('Predicted Salary ($)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    st.pyplot(fig2)

    # Visualization 3: 3D surface or heatmap alternative - Salary by Test & Interview Score
    st.markdown("### Salary Heatmap: Test Score vs Interview Score")
    test_range = np.linspace(0, 10, 21)
    interview_range = np.linspace(0, 10, 21)
    salary_grid = np.zeros((len(test_range), len(interview_range)))
    
    for i, ts in enumerate(test_range):
        for j, its in enumerate(interview_range):
            temp_input = pd.DataFrame({
                'experience': [experience],
                'test_score(out of 10)': [ts],
                'interview_score(out of 10)': [its]
            })
            salary_grid[i, j] = model.predict(temp_input)[0]
    
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    sns.heatmap(salary_grid, xticklabels=np.round(interview_range, 1), yticklabels=np.round(test_range, 1),
                annot=True, fmt=".0f", cmap="YlGnBu", ax=ax3, cbar_kws={'label': 'Predicted Salary ($)'})
    ax3.set_title(f'Predicted Salary Heatmap (Fixed Experience = {experience} years)')
    ax3.set_xlabel('Interview Score')
    ax3.set_ylabel('Test Score')
    st.pyplot(fig3)

# Footer
st.markdown("---")
st.markdown("Model trained on a small hiring dataset for demonstration purposes. "
            "Multi-feature linear regression achieves R² = 0.93.")
