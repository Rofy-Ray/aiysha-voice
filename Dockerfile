FROM python:3.12

WORKDIR /app

COPY . /app

RUN apt update && apt install -y ffmpeg

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install gunicorn

EXPOSE 8080

ENV PORT 8080

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app