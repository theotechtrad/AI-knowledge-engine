FROM python:3.11-slim

RUN apt-get update && apt-get install -y git build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip && pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
COPY clean_reqs.py .

RUN python clean_reqs.py && pip install -r requirements_clean.txt

COPY . .

RUN mkdir -p /data/chroma_db && rm -rf /app/chroma_db && ln -s /data/chroma_db /app/chroma_db

EXPOSE 5500

ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
ENV CHROMA_DB_PATH=/data/chroma_db

CMD ["python", "main.py"]