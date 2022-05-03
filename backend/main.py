import datetime

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
from db.base_class import Base
from db.session import engine
from db.session import get_db

from model import inference

def create_tables():
	print("create_tables")
	Base.metadata.create_all(bind=engine)


app = FastAPI()
create_tables()


@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}


@app.post("/probability")
def get_probability(file: UploadFile = File(...), db: Session = Depends(get_db)):

    price = inference.pricerecognition(file.file)
    label, probability, result = inference.predict(file.file, 0)
    item = {'label': label, "probability": probability, "price": price, "image": result}
    create_new_item(item = item, db=db)

    return {"label": label, "probability": probability, "result": result, "price": price}


@app.get("/get_history", response_model = List)
def read_item(db:Session = Depends(get_db)):
    item = list_item(db=db)
    return item


# @app.post("/get_price")
# def get_price(file: UploadFile = File(...)):
#
#
#     return price



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8090)