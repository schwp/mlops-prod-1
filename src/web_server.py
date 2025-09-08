from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(port=5757)
model = joblib.load("../model/regression.joblib")

class PredictionBody(BaseModel):
    size: float
    nb_rooms: int
    garden: int

@app.get("/predict")
async def get_predict():
    return {"y_pred": 2}

@app.post("/predict")
async def predict_pricing(item: PredictionBody):
    df = pd.DataFrame(
            [[item.size, item.nb_rooms, item.garden]],
            columns=['size', 'nb_rooms', 'garden']
        )
    pred = model.predict(df)
    return {"y_pred": pred[0]}
