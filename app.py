from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import traceback

# Load the saved model
with open('delinquency_model_rf.pkl', 'rb') as f:
    pipeline_data = pickle.load(f)

class DelinquencyPredictorRF:
    def __init__(self, model, scaler, label_encoders, feature_names, task_type):
        self.model = model
        self.scaler = scaler
        self.label_encoders = label_encoders
        self.feature_names = feature_names
        self.task_type = task_type
    
    def predict_single_sample(self, sample_data, threshold=0.25):  # Update if Cell 6.5 differs
        sample_df = pd.DataFrame([sample_data])
        if 'num_late_payments' not in sample_df.columns:
            sample_df['num_late_payments'] = sample_df[['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6']].apply(
                lambda x: (x.isin(['Late', 'Missed'])).sum(), axis=1
            )
        
        for col, encoder in self.label_encoders.items():
            if col in sample_df.columns:
                try:
                    sample_df[col] = encoder.transform(sample_df[col])
                except ValueError as e:
                    raise ValueError(f"Invalid value for {col}: {sample_df[col].values[0]}. Expected one of {encoder.classes_}")
        
        sample_df = pd.get_dummies(sample_df, columns=['Location', 'Credit_Card_Type', 'Employment_Status'], prefix=['Location', 'Card', 'Emp'])
        for col in self.feature_names:
            if col not in sample_df.columns:
                sample_df[col] = 0
        sample_df = sample_df[self.feature_names]
        
        sample_scaled = self.scaler.transform(sample_df)
        prediction = self.model.predict_proba(sample_scaled)[:, 1][0]
        
        return prediction

predictor_rf = DelinquencyPredictorRF(
    model=pipeline_data['model'],
    scaler=pipeline_data['scaler'],
    label_encoders=pipeline_data['label_encoders'],
    feature_names=pipeline_data['feature_names'],
    task_type=pipeline_data['task_type']
)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        required_features = [
            'Age', 'Income', 'Credit_Score', 'Credit_Utilization', 'Missed_Payments',
            'Loan_Balance', 'Debt_to_Income_Ratio', 'Employment_Status', 'Account_Tenure',
            'Credit_Card_Type', 'Location', 'Month_1', 'Month_2', 'Month_3', 'Month_4',
            'Month_5', 'Month_6'
        ]
        missing_features = [f for f in required_features if f not in data]
        if missing_features:
            return jsonify({'error': f'Missing features: {missing_features}'}), 400
        
        pred = predictor_rf.predict_single_sample(data, threshold=0.543)
        return jsonify({
            'probability': float(pred),
            'prediction': 1 if pred > 0.543 else 0
        }), 200
    
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    print("Starting Flask server on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True)