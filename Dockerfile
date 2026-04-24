FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies that ML libraries commonly need
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    # Misc
        build-essential \
        git \
    # Python
        python3-tk \
        python3-opencv \
    # Graphics
        # libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
    # Wayland support
        libwayland-client0 \
        libwayland-egl1 \
    # Qt Wayland plugin for OpenCV GUI
        qtwayland5 \
        libqt5waylandclient5

# Create virtual environment
RUN python -m venv /opt/venv

# Activate venv by modifying PATH
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Create user matching host user (1000:100) for GUI access
RUN useradd -m -u 1000 -g 100 -s /bin/bash user && \
    chown -R 1000:100 /opt/venv /app

# Copy the rest of the application
COPY . .
RUN chown -R 1000:100 /app

# Set entrypoint
# ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Default command - you can override this
CMD ["/bin/bash"]
