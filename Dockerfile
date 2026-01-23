FROM python:3.12

# Prevent interactive prompts during the build
ENV DEBIAN_FRONTEND=noninteractive

# Prevent Python from creating __pycache__ directories
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /rag

COPY requirements.txt /rag/
RUN pip install --no-cache-dir -r /rag/requirements.txt

WORKDIR /rag/app/src

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
