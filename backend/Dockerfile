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

# Make the start script executable
RUN chmod +x start.sh

# Expose port
EXPOSE 8000

# Use the start script instead of just the API server
CMD ["./start.sh"]