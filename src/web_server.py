from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = FastAPI(port=5700)

house_price_model = joblib.load("../model/regression.joblib")

llm_tokenizer = AutoTokenizer.from_pretrained(
    "google-t5/t5-base"
    )
llm_model = AutoModelForSeq2SeqLM.from_pretrained(
    "google-t5/t5-base",
    device_map="auto"
    )

class PredictionBody(BaseModel):
    size: float
    nb_rooms: int
    garden: int

class TranslationBody(BaseModel):
    text: str

@app.get("/predict")
async def get_predict():
    return {"y_pred": 2}

@app.post("/predict")
async def predict_pricing(item: PredictionBody):
    df = pd.DataFrame(
            [[item.size, item.nb_rooms, item.garden]],
            columns=['size', 'nb_rooms', 'garden']
        )
    pred = house_price_model.predict(df)
    return {"y_pred": pred[0]}

@app.post("/translate")
async def translate_text(item: TranslationBody):
    input_ids = llm_tokenizer(
        f"translate French to English: {item.text}",
        return_tensors="pt"
        ).to(llm_model.device)
    output = llm_model.generate(**input_ids, cache_implementation="static")
    return {"translation": llm_tokenizer.decode(output[0], skip_special_tokens=True)}
