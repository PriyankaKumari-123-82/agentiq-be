ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE} as builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    pandoc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN uv pip install --no-cache-dir -e .

ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    binutils \
    bzip2 \
    ca-certificates \
    curl \
    file \
    gzip \
    tar \
    unzip \
    wget \
    libpango-1.0-0 \
    libharfbuzz0b \
    libpangoft2-1.0-0 \
    libfontconfig1 \
    libcairo2 \
    libgdk-pixbuf-xlib-2.0-0 \
    && curl https://rclone.org/install.sh | bash \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN mkdir -p /app/ai_agent/static /app/ai_agent/templates

RUN python -m spacy download en_core_web_md

EXPOSE 8000

CMD ["sh", "-c", "cd /app/ai_agent && alembic -c alembic.ini upgrade heads && cd /app && uvicorn ai_agent.main:app --host 0.0.0.0 --port 8000"]
