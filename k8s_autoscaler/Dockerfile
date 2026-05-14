FROM python:3.10-slim

WORKDIR /app

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=600

# Install heavy ML deps first (large wheels — 600s timeout prevents socket timeout on slow connections)
RUN python -m pip install --no-cache-dir --retries 5 \
    tensorflow==2.17.0 \
    keras==3.4.1 \
    numpy==1.26.4 \
    scikit-learn==1.4.2

# Install lighter runtime dependencies
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --retries 5 -r requirements.txt

# Copy scripts
COPY autoscaler.py .
COPY load_replayer.py .

# Log directory
RUN mkdir -p /var/log

# Model directory (mount your saved model here at runtime)
RUN mkdir -p /model

ENV MODEL_DIR=/model

CMD ["python", "autoscaler.py"]
