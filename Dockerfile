FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY message_finder.py /app/message_finder.py
COPY const.py /app/const.py
COPY config.py /app/config.py
COPY logging_config.py /app/logging_config.py
COPY core /app/core
COPY services /app/services
COPY tg /app/tg
COPY utils /app/utils
COPY utilities /app/utilities
COPY data /app/data
COPY warp_chat.py /app/warp_chat.py

# Ensure runtime directories exist
RUN mkdir -p /app/data /app/results

# Default envs (override in runtime)
ENV LOG_LEVEL=INFO \
    REQUEST_TIMEOUT_S=20 \
    RETRY_MAX_ATTEMPTS=3 \
    RATE_LIMIT_RPM=30 \
    RATE_LIMIT_RPH=900 \
    RATE_LIMIT_TPM=60000 \
    GEMINI_RATE_RPM=15 \
    GEMINI_RATE_TPM=250000 \
    GEMINI_RATE_RPD=1000

# Default to message_finder; override CMD in compose for warp_chat
CMD ["python", "/app/message_finder.py"]

