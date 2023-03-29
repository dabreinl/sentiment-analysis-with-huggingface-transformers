from app.modeling.inference import Inference
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os


class InputTweet(BaseModel):
    tweet: str


config = {
    "model_checkpoint": "distilbert-base-uncased",
    "class_names": ["sadness", "joy", "love", "anger", "fear", "surprise"],
    "saved_model_name": "distilbert-base-finetuned-for-tweet-classification-with-random-oversampling",
}


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    with open("app/static/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/")
async def root():
    return {"health_check": "OK"}


@app.post("/predict")
async def predict(tweet_in: InputTweet):
    tweet = tweet_in.tweet
    inference = Inference(
        config["model_checkpoint"], config["class_names"], config["saved_model_name"]
    )
    sentiment = inference.prediction_pipeline(tweet)
    return JSONResponse(content={"sentiment": sentiment})
