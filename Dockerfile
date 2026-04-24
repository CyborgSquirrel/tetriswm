FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies that ML libraries commonly need
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    build-essential \
    git

# Copy requirements first for better caching
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Default command - you can override this
CMD ["python"]
