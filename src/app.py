import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.pipeline.predict_pipeline import PredictPipeline
from src.logger import logging
from src.exception import CustomException

app = FastAPI()

# IMPORTANT: In production, you should replace "*" with the actual URL of your frontend
# We will get this URL after we deploy the frontend in Step 3.
# For now, we leave it as "*" to make the initial deployment easy.
origins = ["https://diabtes-management.onrender.com"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DiabetesData(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.get('/')
def index():
    return {'message': 'Welcome to the Diabetes Prediction API'}

@app.post('/predict')
def predict_diabetes(data: DiabetesData):
    try:
        logging.info("Received request for prediction.")
        data_dict = data.dict()
        data_df = pd.DataFrame([data_dict])
        
        logging.info("Input data converted to DataFrame:")
        logging.info(data_df.head())

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(data_df)
        
        prediction_value = int(results[0])
        prediction_text = "Positive for Diabetes" if prediction_value == 1 else "Negative for Diabetes"
        logging.info(f"Prediction result: {prediction_text}")

        return {
            "prediction": prediction_text,
            "prediction_value": prediction_value
        }
    except Exception as e:
        logging.error(f"Error occurred during prediction: {e}")
        raise CustomException(e, sys)

# The host '0.0.0.0' makes the app accessible from outside the container
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
