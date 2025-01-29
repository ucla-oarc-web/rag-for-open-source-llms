FROM python:3.10

# Prevent interactive prompts during the build
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /rag

COPY requirements.txt /rag/
RUN pip install --no-cache-dir -r /rag/requirements.txt

WORKDIR /rag/app

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "App:app", "--host", "0.0.0.0", "--port", "8000"]
