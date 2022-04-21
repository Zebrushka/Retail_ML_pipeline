# frontend/Dockerfile

FROM python:3.10.1-slim

WORKDIR /app

RUN apt-get install gcc python3-dev

COPY requirements.txt .

RUN pip install -r requirements.txt


COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py"]
