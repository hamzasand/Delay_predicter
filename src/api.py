# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
import pandas as pd
import joblib
import os

# Load trained model and feature engineer
MODEL_PATH = 'models/xgboost_model.pkl'
FEATURE_ENGINEER_PATH = 'models/feature_engineer.pkl'

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_ENGINEER_PATH):
    raise FileNotFoundError("Model artifacts not found. Please train the model first.")

model = joblib.load(MODEL_PATH)
feature_engineer = joblib.load(FEATURE_ENGINEER_PATH)

app = FastAPI(
    title="Construction Delay Predictor API",
    description="API for predicting construction task delays",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    planned_start_date: str = Field(..., example="2025-01-10")
    planned_end_date: str = Field(..., example="2025-01-20")
    crew_name: str = Field(..., example="masonry_crew")
    task_label: str = Field(..., example="blockwork walls")
    region: str = Field(..., example="lahore")
    project_id: int = Field(default=999, example=999)
    task_id: int = Field(default=1, example=1)

class PredictionResponse(BaseModel):
    predicted_delay_days: float = Field(..., example=6.8)

@app.get("/")
def read_root():
    return {
        "message": "Construction Delay Predictor API",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_delay(request: PredictionRequest):
    """
    Predict construction task delay in days
    """
    try:
        # Parse dates
        planned_start = pd.to_datetime(request.planned_start_date)
        planned_end = pd.to_datetime(request.planned_end_date)
        
        # Calculate planned duration
        planned_duration = (planned_end - planned_start).days + 1
        
        if planned_duration <= 0:
            raise HTTPException(
                status_code=400, 
                detail="planned_end_date must be after planned_start_date"
            )
        
        # Create input dataframe
        input_data = pd.DataFrame([{
            'project_id': request.project_id,
            'task_id': request.task_id,
            'task_label': request.task_label,
            'crew_name': request.crew_name,
            'region': request.region,
            'planned_start_date': planned_start,
            'planned_end_date': planned_end,
            'planned_duration_days': planned_duration
        }])
        
        # Apply feature engineering (fit=False to use existing encoders)
        input_data = feature_engineer.create_temporal_features(input_data)
        input_data = feature_engineer.create_duration_features(input_data)
        
        # Set default values for task complexity features
        input_data['task_sequence'] = 1
        input_data['project_total_tasks'] = 5
        
        input_data = feature_engineer.encode_categorical_features(input_data, fit=False)
        input_data = feature_engineer.create_interaction_features(input_data)
        
        # Select features
        X = input_data[feature_engineer.feature_columns]
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        return PredictionResponse(predicted_delay_days=round(float(prediction), 2))
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)