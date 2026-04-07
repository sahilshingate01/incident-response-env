FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

EXPOSE 7860

CMD ["uvicorn", "src.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
