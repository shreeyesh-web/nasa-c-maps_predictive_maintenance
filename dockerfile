# Base Python image - slim keeps it small
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies PostgreSQL needs
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker caches this layer)
# If requirements don't change, Docker skips reinstalling
COPY requirements.txt .

# Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into container
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to start FastAPI when container starts
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]