FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY docs/ ./docs/
COPY .env .

WORKDIR /app/backend

# Run ingestion first, then start API
CMD ["sh", "-c", "python ingest.py && uvicorn main:app --host 0.0.0.0 --port 8000"]