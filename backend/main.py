import uvicorn
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
import cv2
import uuid

from model import inference, priceRecognition

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}


@app.post("/probability")
def get_image(file: UploadFile = File(...)):

    label, probability, result = inference.predict(file.file)
    price = inference.priceRecognition(file.file, 1)

    return {"label": label, "probability": probability, "result": result, "price": price}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8090)