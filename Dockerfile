FROM python:3.8

WORKDIR /app
COPY . .
COPY requirements.txt requirements.txt

RUN apt-get update -y
RUN apt install -y vim
RUN apt-get install libsndfile1 -y

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8000
ENTRYPOINT ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
