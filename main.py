from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn
import pandas as pd
from predict_traffic import TrafficPred
from fastapi.middleware.cors import CORSMiddleware

# Define the input data schema
class TrafficInput(BaseModel):
    lag1: int
    lag2: int

# Create the FastAPI app
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

origins = [
    "https://traffic-predictor-vep6-k98cfs09y-codeairs-projects.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # only your site allowed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Prediction endpoint
@app.post("/predict")
def predict_traffic(data: TrafficInput):
    prediction = TrafficPred(data.lag1, data.lag2)
    return {"predicted_traffic_volume": float(prediction)}

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
