import uvicorn
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
from PIL import Image

from model import inference

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}


@app.post("/probability")
def get_image(file: UploadFile = File(...)):
    # image = Image.open(file.file)
    label, probability = inference.predict(file.file)
    return {"label": label, "probability": probability}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8090)