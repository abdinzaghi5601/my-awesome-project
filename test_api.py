import requests

print("Starting API tests...")
url = 'http://127.0.0.1:5000/predict'

data = {
    "Age": 35,
    "Income": 50000,
    "Credit_Score": 650,
    "Credit_Utilization": 0.3,
    "Missed_Payments": 0,
    "Loan_Balance": 15000,
    "Debt_to_Income_Ratio": 0.3,
    "Employment_Status": "Employed",
    "Account_Tenure": 5,
    "Credit_Card_Type": "Standard",
    "Location": "Chicago",
    "Month_1": "On-time",
    "Month_2": "On-time",
    "Month_3": "On-time",
    "Month_4": "On-time",
    "Month_5": "On-time",
    "Month_6": "On-time"
}

try:
    print("Sending low-risk request...")
    response = requests.post(url, json=data)
    print("Low-Risk Response:", response.json())
except Exception as e:
    print("Low-Risk Error:", str(e))

high_risk_data = {
    "Age": 30,
    "Income": 30000,
    "Credit_Score": 500,
    "Credit_Utilization": 0.9,
    "Missed_Payments": 3,
    "Loan_Balance": 25000,
    "Debt_to_Income_Ratio": 0.6,
    "Employment_Status": "Unemployed",
    "Account_Tenure": 2,
    "Credit_Card_Type": "Basic",
    "Location": "New_York",
    "Month_1": "Missed",
    "Month_2": "Late",
    "Month_3": "On-time",
    "Month_4": "Missed",
    "Month_5": "Late",
    "Month_6": "Missed"
}

try:
    print("Sending high-risk request...")
    response = requests.post(url, json=high_risk_data)
    print("High-Risk Response:", response.json())
except Exception as e:
    print("High-Risk Error:", str(e))