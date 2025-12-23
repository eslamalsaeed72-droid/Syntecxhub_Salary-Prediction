# Salary Prediction Model using Linear Regression

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/stable/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.0%2B-red.svg)](https://streamlit.io/)
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-Orange.svg)](https://jupyter.org/)

## Project Overview

This repository contains a complete machine learning project for predicting candidate salaries based on years of experience, written test scores, and interview performance. The implementation includes data preprocessing, exploratory data analysis (EDA), model training and evaluation using linear regression, and an interactive web application built with Streamlit.

Key highlights:
- Comparison of single-feature and multi-feature linear regression models.
- Evaluation metrics: RMSE and R² score.
- Best model performance: R² = 0.93, RMSE ≈ $2,213 on the test set.
- Deployable Streamlit app for real-time predictions.

## Dataset

- **Source**: Hiring Dataset from [Kaggle](https://www.kaggle.com/datasets/anaghakp/hiring-dataset).
- **Features**:
  - `experience`: Years of professional experience (encoded from categorical text).
  - `test_score(out of 10)`: Written test score.
  - `interview_score(out of 10)`: Interview performance score.
- **Target**: `salary($)`.

Preprocessing steps handle missing values and categorical encoding.

## Key Components

1. **Exploratory Data Analysis (EDA)**: Visualizations of data distributions and feature-target relationships.
2. **Data Preprocessing**: Handling missing values, encoding categorical features.
3. **Model Training**: Single-feature models for each input variable and a multi-feature model.
4. **Evaluation**: Comparison using RMSE and R² on a test set.
5. **Model Persistence**: Best model saved as `best_model.pkl`.
6. **Web Application**: Interactive Streamlit app for user inputs and predictions.

## Requirements

Install dependencies via:
```bash
pip install -r requirements.txt
```

`requirements.txt` contents:
```
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
streamlit
```

## Usage Instructions

### Running the Notebook
1. Open `Syntecxhub_Salary Prediction..ipynb` in Jupyter Notebook or Google Colab.
2. Execute cells sequentially to perform EDA, train models, and save the best model.

### Running the Streamlit App Locally
```bash
streamlit run app.py
```
- Use sliders to input values and predict salary.

### Deployment
- Deploy the Streamlit app on platforms like Streamlit Community Cloud by linking this GitHub repository.
- Ensure `best_model.pkl` and `requirements.txt` are included in the deployment.

## Model Performance Summary

| Model Type             | RMSE       | R² Score |
|------------------------|------------|----------|
| Single (Experience)    | 6,238.45   | 0.46     |
| Single (Test Score)    | 15,254.33  | -2.22    |
| Single (Interview Score)| 18,072.26 | -3.52    |
| Multiple Features      | 2,213.28   | 0.93     |

The multi-feature model is selected as the best performer and used in the application.

## Future Enhancements
- Incorporate larger datasets for improved generalization.
- Explore advanced regression techniques (e.g., Ridge, Lasso).
- Implement cross-validation and hyperparameter tuning.
- Add unit tests for code reliability.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome. Please submit pull requests or open issues for suggestions.

## Contact

For questions or collaboration, reach out via GitHub issues.
email:eslamalsaeed72@gmail.com
