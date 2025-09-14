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

COPY message_finder.py /app/message_finder.py
COPY const.py /app/const.py
COPY utilities /app/utilities
COPY data /app/data

# Default envs (override in runtime)
ENV LOG_LEVEL=INFO \
    REQUEST_TIMEOUT_S=20 \
    RETRY_MAX_ATTEMPTS=3 \
    RATE_LIMIT_RPM=30 \
    RATE_LIMIT_RPH=900 \
    RATE_LIMIT_TPM=60000

CMD ["python", "/app/message_finder.py"]

