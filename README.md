# Retail ML pipeline

![Пример работы](image.png)

A project on machine learning courses. Pipeline is used internally: YOLO for item detection, price tags and price, EfficientNet for item classification. FastAPI and Streamlit as backend and frontend respectively.

Presentation about project: ![link](https://docs.google.com/presentation/d/1PB2b-QVpy2HFX47W4n3bQU0znsZWkqkC3RoWxc9yvcs/edit#slide=id.g127fd7df2fd_0_7)

Demo: http://62.148.235.120:8501/

Build: `docker-compose up -d --build`

Run: `docker-compose up -d`


Already implemented and plans:
- [x] YOLO for item detection, price tags and price
- [x] EfficientNet for item classification
- [x] Backend + frontend
- [x] For unknown products display the label unknown
- [x] OCR
- [x] Save history to database (sqlite)
