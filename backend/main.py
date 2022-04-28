import uvicorn
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
import cv2
import uuid

from model import inference

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}


@app.post("/probability")
def get_image(file: UploadFile = File(...)):
    # image = Image.open(file.file)
    label, probability, result = inference.predict(file.file)
    name = f"/storage/{str(uuid.uuid4())}.jpg"
    isWritten = cv2.imwrite(name, result)
    if isWritten:
        print('Image is successfully saved as file to path: ', name)
    return {"label": label, "probability": probability, "result": name}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8090)