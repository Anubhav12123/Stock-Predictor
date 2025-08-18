FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir poetry
WORKDIR /app
COPY pyproject.toml README.md .
RUN poetry install --no-root --no-interaction --no-ansi
COPY src ./src
EXPOSE 8000
CMD ["poetry","run","uvicorn","src.api.main:app","--host","0.0.0.0","--port","8000","--workers","2"]
