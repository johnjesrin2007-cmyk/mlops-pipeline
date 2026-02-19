FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only requirements first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY main.py .
COPY model/ ./model/

# If you need src folder for imports
COPY src/ ./src/

# Expose port (optional but good practice)
EXPOSE 8000

# Start FastAPI app
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
