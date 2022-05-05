import datetime

import requests
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
def get_probability(file: UploadFile = File(...)):

    label, probability, result = inference.predict(file.file, 0)

    return {"label": label, "probability": probability, "result": result}


@app.get("/get_history", response_model = List)
def read_item(db:Session = Depends(get_db)):
    item = list_item(db=db)
    return item

@app.post("/get_price")
def read_item(file: UploadFile = File(...)):
    price = inference.pricerecognition(file.file)
    return {"price" : price[0]}

@app.post("/write_db")
def write_db(item, db: Session = Depends(get_db)):
    # item = {'label': item[0], "probability": item[1], "price": item[2], "image": item[3]}
    print(item)
    create_new_item(item = item, db=db)



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8090)