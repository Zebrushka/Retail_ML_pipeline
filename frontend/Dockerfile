# frontend/Dockerfile

FROM python:3.10.1-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install --no-binary :all: psutil

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py"]
