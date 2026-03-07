FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends     git curl ca-certificates bash     && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir mlx-lm

WORKDIR /app
COPY mlx /app/mlx

EXPOSE 8081
