# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a **delinquency prediction system** for financial institutions that uses machine learning to predict customer account delinquency. The project implements multiple ML approaches (Random Forest, Logistic Regression, Neural Networks) with a production-ready Flask API.

## Common Commands

### Running the Application
```bash
# Start the Flask API server
python app.py

# Run API tests
python test_api.py
```

### Development Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Launch Jupyter notebooks for model development
jupyter notebook
```

### Data Processing
```bash
# Execute database setup and data cleaning
mysql -u root -p < tata.sql
```

## Architecture Overview

### Core Components
1. **Flask API** (`app.py`) - Production REST API using Random Forest model
2. **ML Models** - Three approaches with different performance characteristics:
   - **Random Forest** (Production): AUC 0.97, F1 0.86 - Best performing
   - **Neural Network** (Experimental): AUC 0.59, F1 0.27 - Deep learning approach  
   - **Logistic Regression** (Baseline): AUC 0.48, F1 0.27 - Interpretable baseline
3. **Database Layer** (`tata.sql`) - MySQL schema with data cleaning procedures
4. **Testing Suite** (`test_api.py`) - API endpoint validation

### Data Flow Architecture
```
CSV Dataset → Feature Engineering → Model Training → Pickle Serialization → Flask API → Predictions
```

### Model Pipeline
- **Input**: 17 features (demographics, credit profile, payment history)
- **Feature Engineering**: Advanced features (num_late_payments, credit_score_dti, dti_squared)
- **Class Imbalance**: SMOTE handling for 16% delinquency rate
- **Threshold Optimization**: Custom threshold (0.543 for Random Forest)
- **Output**: Probability score + binary classification

## Key Dependencies
- **Core ML**: pandas, numpy, scikit-learn, matplotlib, seaborn
- **Class Balance**: imbalanced-learn (SMOTE)
- **Deep Learning**: tensorflow, shap (explainability)
- **API**: Flask web framework
- **Database**: MySQL (TATABANK)

## API Specification

### Prediction Endpoint
```
POST /predict
Content-Type: application/json

Required Fields (17):
- Age, Income, Credit_Score, Credit_Utilization, Missed_Payments
- Loan_Balance, Debt_to_Income_Ratio, Employment_Status, Account_Tenure
- Credit_Card_Type, Location
- Month_1 through Month_6 (payment history: "On-time"/"Late"/"Missed")

Response:
{
  "prediction": 0|1,
  "probability": 0.0-1.0
}
```

## Model Performance Hierarchy
1. **Random Forest** (Production Model): `delinquency_model_rf.pkl`
2. **Neural Network** (Experimental): `best_model.h5` + `delinquency_nn_pipeline.joblib`
3. **Logistic Regression** (Baseline): `delinquency_model_lr.pkl`

## Data Schema
- **Dataset**: 500 customers × 20 features
- **Target Distribution**: 420 non-delinquent, 80 delinquent (16% positive class)
- **Missing Values**: Handled in preprocessing pipeline
- **Categorical Encoding**: One-hot for nominal, label for ordinal features

## Testing Strategy
- **API Testing**: Realistic customer scenarios (low-risk vs high-risk profiles)
- **Model Validation**: Cross-validation with stratified splits
- **Performance Metrics**: AUC-ROC, Precision, Recall, F1-Score prioritized
- **Threshold Tuning**: Optimized for business objectives

## File Dependencies
```
app.py → delinquency_model_rf.pkl (trained model)
All notebooks → Delinquency_prediction_dataset.csv (shared dataset)
tata.sql → Data cleaning and schema setup
test_api.py → app.py (API testing)
```

## Development Notes
- **Production Model**: Always use Random Forest for highest accuracy
- **Feature Engineering**: Critical for model performance - maintain consistency between training and inference
- **Class Imbalance**: 16% delinquency rate requires careful handling (SMOTE implemented)
- **Threshold Selection**: 0.25 (API default) vs 0.543 (optimized) - consider business requirements
- **Model Monitoring**: Track prediction distribution and feature drift in production