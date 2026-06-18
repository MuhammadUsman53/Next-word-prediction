# app.py

from fastapi import FastAPI
from pydantic import BaseModel
from backend import predict_next_word

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(data: TextInput):
    result = predict_next_word(data.text)
    return {"next_word": result}
