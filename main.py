from fastapi import FastAPI
from modeling.inference import Inference
from pydantic import BaseModel

config = {
    "model_checkpoint": "distilbert-base-uncased",
    "class_names": ["sadness", "joy", "love", "anger", "fear", "surprise"],
    "saved_model_name": "distilbert-base-finetuned-for-tweet-classification",
}

app = FastAPI()


@app.get("/")
async def root():
    return {"health_check": "OK"}


@app.post("/predict")
async def predict(TweetIn: str):
    inference = Inference(
        config["model_checkpoint"], config["class_names"], config["saved_model_name"]
    )
    sentiment = inference.prediction_pipeline(TweetIn)
    return {"sentiment": sentiment}
