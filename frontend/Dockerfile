# frontend/Dockerfile

FROM python:3.10.1-slim

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install psutil
RUN pip install -r requirements.txt


COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py"]
