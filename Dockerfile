FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# This copies everything, including your 'mlruns' folder
COPY . .
# Start the app on the port Render provides
CMD uvicorn main:app --host 0.0.0.0 --port $PORT