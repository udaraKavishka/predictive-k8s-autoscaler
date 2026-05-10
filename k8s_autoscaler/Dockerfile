FROM tensorflow/tensorflow:2.17.0

WORKDIR /app

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONUNBUFFERED=1

# Install Python deps first for layer caching
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy scripts
COPY autoscaler.py .
COPY load_replayer.py .

# Log directory
RUN mkdir -p /var/log

# Model directory (mount your saved model here at runtime)
RUN mkdir -p /model

ENV MODEL_DIR=/model

CMD ["python", "autoscaler.py"]
