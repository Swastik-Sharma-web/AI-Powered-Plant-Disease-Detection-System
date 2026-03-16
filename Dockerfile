# Use the official Python 3.10 image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies and add OpenCV for the leaf detector
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir opencv-python-headless

# Copy the remaining project files
# We need backend, models, and training folders
COPY backend/ backend/
COPY models/ models/
COPY training/ training/

# Hugging Face Spaces uses port 7860 by default
EXPOSE 7860

# Run the FastAPI server on port 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
