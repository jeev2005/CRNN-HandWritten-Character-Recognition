FROM python:3.9-slim

# Install system dependencies for OpenCV and EasyOCR
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user to avoid running as root
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download EasyOCR models (Hindi and English)
# This prevents the "Starting" hang and avoids runtime DNS issues.
RUN python -c "import easyocr; reader = easyocr.Reader(['hi', 'en'], download_enabled=True)"

# Copy the rest of the project
COPY --chown=user . .

# Set environment variables
ENV FLASK_APP=server.py \
    PYTHONUNBUFFERED=1

# Hugging Face Spaces uses port 7860 by default
EXPOSE 7860

# Run the server
CMD ["python", "server.py"]
