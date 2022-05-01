import uvicorn
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import Depends

import cv2
import uuid
from typing import List

from db.session import get_db
from db.models.item import Item
from sqlalchemy.orm import Session
from db.repository.item import create_new_item, list_item

from model import inference

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}


@app.post("/probability")
def get_probability(file: UploadFile = File(...)):

    label, probability, result = inference.predict(file.file, 0)
    price = inference.priceRecognition(file.file)
    item = [label, probability, price, result]
    create_new_item(item)

    return {"label": label, "probability": probability, "result": result, "price": price}


@app.get("/get_history", response_model = List)
def read_item(db:Session = Depends(get_db)):
    item = list_item(db=db)
    return item



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8090)