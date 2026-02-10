FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Point to api.main:app because main.py is inside the api/ folder
# This tells uvicorn to look for 'app' inside 'main.py' in the current directory
CMD ["uvicorn main:app --host 0.0.0.0 --port $PORT"]