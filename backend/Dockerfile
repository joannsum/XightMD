FROM python:3.12.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy minimal requirements for deployment
COPY requirements-minimal.txt .

# Install Python dependencies with no cache to save space
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-minimal.txt

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 8000

# Start the FastAPI server
CMD ["python", "api/server.py"]