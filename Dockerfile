FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[anthropic]"

COPY sources.yaml config.example.yaml ./
RUN useradd --create-home --uid 10001 app && mkdir -p /app/data && chown app /app/data
USER app

# Offline demo by default; for real feeds:
#   docker run --rm -v $PWD/data:/app/data ai-news-pipeline run --config config.example.yaml
ENTRYPOINT ["ai-news-pipeline"]
CMD ["demo", "--out", "/app/data/demo"]
