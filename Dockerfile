FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Point to api.main:app because main.py is inside the api/ folder
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]