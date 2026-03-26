FROM python:3.13-slim

ARG PIPELINE_BUILD=dev

WORKDIR /app/src

RUN apt-get update && rm -rf /var/lib/apt/lists/*

COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

ENV PIPELINE_BUILD=${PIPELINE_BUILD}

RUN useradd --create-home --shell /bin/bash app
USER app

EXPOSE 8080
CMD ["python", "main_service.py"]
