"""
FastAPI deployment for delinquency prediction model
Production-ready with automatic documentation, validation, and monitoring
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import pickle
import pandas as pd
import numpy as np
from typing import List, Optional
import uvicorn
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
try:
    with open('delinquency_model_rf.pkl', 'rb') as f:
        pipeline_data = pickle.load(f)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

app = FastAPI(
    title="Delinquency Prediction API",
    description="Machine learning API for predicting customer account delinquency",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

class CustomerData(BaseModel):
    """Input data model with validation"""
    Age: int
    Income: float
    Credit_Score: int
    Credit_Utilization: float
    Missed_Payments: int
    Loan_Balance: float
    Debt_to_Income_Ratio: float
    Employment_Status: str
    Account_Tenure: int
    Credit_Card_Type: str
    Location: str
    Month_1: str
    Month_2: str
    Month_3: str
    Month_4: str
    Month_5: str
    Month_6: str
    
    @validator('Credit_Score')
    def validate_credit_score(cls, v):
        if not 300 <= v <= 850:
            raise ValueError('Credit score must be between 300 and 850')
        return v
    
    @validator('Credit_Utilization', 'Debt_to_Income_Ratio')
    def validate_ratios(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Ratios must be between 0 and 1')
        return v
    
    @validator('Employment_Status')
    def validate_employment(cls, v):
        valid_statuses = ['employed', 'unemployed', 'self_employed', 'retired']
        if v.lower() not in valid_statuses:
            raise ValueError(f'Employment status must be one of: {valid_statuses}')
        return v.lower()
    
    @validator('Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6')
    def validate_payment_history(cls, v):
        valid_payments = ['On-time', 'Late', 'Missed']
        if v not in valid_payments:
            raise ValueError(f'Payment status must be one of: {valid_payments}')
        return v

class PredictionResponse(BaseModel):
    """Response model"""
    probability: float
    prediction: int
    risk_level: str
    confidence: str

class DelinquencyPredictorRF:
    def __init__(self, model, scaler, label_encoders, feature_names, task_type):
        self.model = model
        self.scaler = scaler
        self.label_encoders = label_encoders
        self.feature_names = feature_names
        self.task_type = task_type
    
    def predict_single_sample(self, sample_data: dict, threshold: float = 0.543):
        try:
            sample_df = pd.DataFrame([sample_data])
            
            # Feature engineering
            if 'num_late_payments' not in sample_df.columns:
                sample_df['num_late_payments'] = sample_df[['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6']].apply(
                    lambda x: ((x == 'Late').sum() + 2 * (x == 'Missed').sum()), axis=1
                )
            
            if 'recent_late_payments' not in sample_df.columns:
                sample_df['recent_late_payments'] = sample_df[['Month_4', 'Month_5', 'Month_6']].apply(
                    lambda x: ((x == 'Late').sum() + 2 * (x == 'Missed').sum()), axis=1
                )
            
            if 'credit_score_dti' not in sample_df.columns:
                sample_df['credit_score_dti'] = sample_df['Credit_Score'] * sample_df['Debt_to_Income_Ratio']
                # Apply same transformation as training
                sample_df['credit_score_dti'] = 1 - ((sample_df['credit_score_dti'] - sample_df['credit_score_dti'].min()) / 
                                                    (sample_df['credit_score_dti'].max() - sample_df['credit_score_dti'].min()))
            
            if 'dti_squared' not in sample_df.columns:
                sample_df['dti_squared'] = sample_df['Debt_to_Income_Ratio'] ** 2
            
            # Drop month columns after feature engineering
            sample_df.drop(columns=[f'Month_{i}' for i in range(1, 7)], inplace=True, errors='ignore')
            
            # One-hot encoding
            sample_df = pd.get_dummies(sample_df, columns=['Location', 'Credit_Card_Type', 'Employment_Status'], 
                                      prefix=['Location', 'Card', 'Emp'])
            
            # Align features with training data
            for col in self.feature_names:
                if col not in sample_df.columns:
                    sample_df[col] = 0
            sample_df = sample_df[self.feature_names]
            
            # Scale and predict
            sample_scaled = self.scaler.transform(sample_df)
            probability = self.model.predict_proba(sample_scaled)[:, 1][0]
            
            return float(probability)
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Initialize predictor
predictor_rf = DelinquencyPredictorRF(
    model=pipeline_data['model'],
    scaler=pipeline_data['scaler'],
    label_encoders=pipeline_data['label_encoders'],
    feature_names=pipeline_data['feature_names'],
    task_type=pipeline_data['task_type']
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Delinquency Prediction API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": predictor_rf.model is not None,
        "features_count": len(predictor_rf.feature_names)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_delinquency(customer: CustomerData):
    """
    Predict customer delinquency risk
    
    Returns:
    - probability: Risk probability (0-1)
    - prediction: Binary classification (0=Low Risk, 1=High Risk)
    - risk_level: Human-readable risk level
    - confidence: Model confidence level
    """
    try:
        # Convert to dict
        customer_dict = customer.dict()
        
        # Make prediction
        probability = predictor_rf.predict_single_sample(customer_dict, threshold=0.543)
        prediction = 1 if probability > 0.543 else 0
        
        # Determine risk level and confidence
        if probability < 0.3:
            risk_level = "Low"
            confidence = "High"
        elif probability < 0.7:
            risk_level = "Medium" if prediction == 1 else "Low"
            confidence = "Medium"
        else:
            risk_level = "High"
            confidence = "High"
        
        logger.info(f"Prediction made: probability={probability:.4f}, prediction={prediction}")
        
        return PredictionResponse(
            probability=probability,
            prediction=prediction,
            risk_level=risk_level,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    """Get model information"""
    return {
        "model_type": "RandomForestClassifier",
        "features_count": len(predictor_rf.feature_names),
        "features": predictor_rf.feature_names,
        "threshold": 0.543,
        "performance": {
            "auc_roc": 0.9715,
            "precision": 0.8824,
            "recall": 0.8333,
            "f1_score": 0.8571
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)