FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# No folder prefix needed since main.py is in the root
CMD uvicorn main:app --host 0.0.0.0 --port $PORT