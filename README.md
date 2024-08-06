# Skeletal Muscle Density in Ovarian-Cancer

## Description
Use machine learning to predict whether ovarian cancer patients have muscle radiodensity loss, and use SHAP to explain the prediction results.

## Usage
### handover_ml_shap.py
Perform modeling and prediction, and use SHAP to explain the prediction results.
- **Data Preprocessing**: Includes steps for data cleaning, standardization, and feature engineering.
- **Modeling**: Uses various machine learning models (CatBoostClassifier, RandomForestClassifier, SVC, XGBClassifier).
- **Bootstrap Training**: Implements bootstrap training to ensure model robustness.
- **Model Validation**: Provides methods for validating the model performance.
- **SHAP Explanations**: Generates SHAP summary plot, dependence plot, and force plot to explain the prediction results.
- **Note**: Since the analysis data consists of hospital patient data, it is not provided. To analyze your own data, replace the data in the script with your own dataset.

### handover_roc_plot.py
Present the results after bootstrapping using ROC curves.
- **Data Preprocessing**: Includes methods for cleaning and preparing data for ROC analysis.
- **Plotting**: Provides functions to generate ROC plots for visualizing model performance.

## Execution environment
This study was conducted using Python 3.9. The machine learning models were implemented using Scikit-learn version 1.4.1. The following model packages from the Scikit-learn library were used: CatBoostClassifier, RandomForestClassifier, SVC, and XGBClassifier. Feature contributions were calculated using SHAP version 0.44.1, and visualizations were generated using SHAP and matplotlib version 3.8.3.

## Support
Email: jiamin4010@gmail.com 
Lin, Sally
