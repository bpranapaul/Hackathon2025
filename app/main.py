from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from app.model_utils import predict_prob, predict_class, features

app = FastAPI()

class PredictionRequest(BaseModel):
    instances: list[list[float]]

@app.get("/")
def health():
    return {"message": "Model is running"}

@app.post("/predict")
def predict(req: PredictionRequest):
    try:
        predictions = []
        probabilities = []

        for instance in req.instances:
            input_dict = dict(zip(features, instance))
            prob = predict_prob(input_dict)
            pred = predict_class(prob)

            probabilities.append(prob)
            predictions.append(pred)

        return {
            "messages": [
                "Prediction successful",
                "Predicts the probability of class 1 (cart abandonment) or class 0 (cart purchase)",
                "Define threshold for classifying high abandonment risk: threshold = 0.65"
            ],
            "predictions": predictions, 
            "probabilities": probabilities
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
