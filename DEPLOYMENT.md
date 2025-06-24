# Deployment Guide for Delinquency Prediction Model

## 1. Local Development

### Flask (Simple)
```bash
# Install dependencies
pip install flask pandas scikit-learn numpy

# Run locally
python app.py
# Access at: http://localhost:5000
```

### FastAPI (Recommended)
```bash
# Install dependencies
pip install fastapi uvicorn pandas scikit-learn numpy pydantic

# Run locally
python fastapi_app.py
# Access at: http://localhost:8000
# API docs at: http://localhost:8000/docs
```

## 2. Docker Deployment

### Build and Run
```bash
# Build Docker image
docker build -t delinquency-api .

# Run container
docker run -p 5000:5000 delinquency-api

# Or use docker-compose
docker-compose up --build
```

### Docker Hub Deployment
```bash
# Tag and push to Docker Hub
docker tag delinquency-api your-username/delinquency-api:latest
docker push your-username/delinquency-api:latest
```

## 3. Cloud Deployments

### AWS Lambda
1. Install AWS CLI and configure credentials
2. Create deployment package:
   ```bash
   pip install -r requirements.txt -t ./lambda_package
   cp deploy_aws.py lambda_package/
   cp delinquency_model_rf.pkl lambda_package/
   cd lambda_package && zip -r ../lambda_function.zip .
   ```
3. Upload to AWS Lambda or use AWS SAM/CDK

### AWS ECS/Fargate
```bash
# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker tag delinquency-api:latest <account>.dkr.ecr.us-east-1.amazonaws.com/delinquency-api:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/delinquency-api:latest
```

### Heroku
```bash
# Install Heroku CLI
heroku create delinquency-prediction-api
heroku container:push web
heroku container:release web
```

### Google Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/delinquency-api
gcloud run deploy --image gcr.io/PROJECT_ID/delinquency-api --platform managed
```

### Azure Container Instances
```bash
# Deploy to Azure
az container create --resource-group myResourceGroup --name delinquency-api --image your-registry/delinquency-api:latest --ports 5000
```

## 4. API Usage Examples

### FastAPI Request (Recommended)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 35,
    "Income": 50000,
    "Credit_Score": 650,
    "Credit_Utilization": 0.3,
    "Missed_Payments": 0,
    "Loan_Balance": 15000,
    "Debt_to_Income_Ratio": 0.3,
    "Employment_Status": "employed",
    "Account_Tenure": 5,
    "Credit_Card_Type": "Standard",
    "Location": "Chicago",
    "Month_1": "On-time",
    "Month_2": "On-time",
    "Month_3": "On-time",
    "Month_4": "On-time",
    "Month_5": "On-time",
    "Month_6": "On-time"
  }'
```

### Flask Request
```bash
curl -X POST "http://localhost:5000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 35,
    "Income": 50000,
    "Credit_Score": 650,
    "Credit_Utilization": 0.3,
    "Missed_Payments": 0,
    "Loan_Balance": 15000,
    "Debt_to_Income_Ratio": 0.3,
    "Employment_Status": "employed",
    "Account_Tenure": 5,
    "Credit_Card_Type": "Standard",
    "Location": "Chicago",
    "Month_1": "On-time",
    "Month_2": "On-time",
    "Month_3": "On-time",
    "Month_4": "On-time",
    "Month_5": "On-time",
    "Month_6": "On-time"
  }'
```

## 5. Production Considerations

### Security
- Add API authentication (JWT, API keys)
- Use HTTPS/TLS encryption
- Implement rate limiting
- Add input sanitization

### Monitoring
- Add logging and metrics collection
- Set up health checks
- Monitor model performance drift
- Implement alerting

### Scaling
- Use load balancers
- Implement auto-scaling
- Consider caching for frequent requests
- Use CDN for static assets

### Model Management
- Version control for models
- A/B testing framework
- Model retraining pipeline
- Rollback capabilities

## 6. Environment Variables

Create `.env` file:
```
MODEL_PATH=delinquency_model_rf.pkl
THRESHOLD=0.543
LOG_LEVEL=INFO
ENVIRONMENT=production
```

## 7. Performance Benchmarks

- **Model Size**: ~50MB
- **Cold Start**: <2 seconds
- **Prediction Time**: <100ms
- **Memory Usage**: ~200MB
- **Concurrent Requests**: 100+ (with proper scaling)

## 8. Next Steps

1. Choose deployment method based on requirements
2. Set up CI/CD pipeline
3. Implement monitoring and alerting
4. Plan model update strategy
5. Consider batch prediction capabilities