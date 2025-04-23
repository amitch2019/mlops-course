from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import numpy as np
import uvicorn
import os

app = FastAPI(
    title="Wine Classification API",
    description="API for classifying wine samples based on their features",
    version="0.1",
)

# Define the request body schema
class WineFeatures(BaseModel):
    features: list[float]

@app.on_event('startup')
def load_model():
    global model
    # Use the direct path to the model rather than the registry
    model_path = "mlruns/0/38eeccbcf4364cebbb15f25c76705133/artifacts/model"
    try:
        model = mlflow.pyfunc.load_model(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        # We'll check if the model is available when handling requests

@app.get('/')
def home():
    return {'message': 'Wine Classification Model API'}

@app.post('/predict')
def predict(data: WineFeatures):
    try:
        # Check if model is loaded
        if 'model' not in globals():
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Convert features to numpy array
        features = np.array(data.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        
        # Return prediction
        return {
            'prediction': int(prediction[0]),
            'class_name': ['class_0', 'class_1', 'class_2'][int(prediction[0])],
            'prediction_probabilities': model.predict_proba(features).tolist()[0] if hasattr(model, 'predict_proba') else None
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)