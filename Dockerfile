FROM python:3.11-slim

ARG UID
ARG GID

RUN test -n "$UID" || (echo "UID build arg is required" && exit 1)
RUN test -n "$GID" || (echo "GID build arg is required" && exit 1)

# Set working directory
WORKDIR /app

# Install system dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    # Misc
        build-essential \
        git \
        swig \
        neovim \
    # Python
        python3-tk \
        python3-opencv \
        python3-pil \
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

# Create user matching host user
RUN useradd -m -u "$UID" -g "$GID" -s /bin/bash user
RUN mkdir -p /home/user && chown "$UID:$GID" /home/user

USER $UID:$GID

RUN mkdir -p /home/user/.cache

# Install uv
RUN --mount=type=cache,uid=$UID,gid=$GID,target=/home/user/.cache/pip \
    pip install uv

ENV UV_PYTHON_CACHE_DIR=/home/user/.cache/uv

# Create virtual environment
RUN python -m venv /home/user/venv

# RUN stat /home/user && stat /home/user/.cache

ENV PATH="/home/user/.local/bin:$PATH"

# Activate venv by modifying PATH
ENV PATH="/home/user/venv/bin:$PATH"

# Copy requirements first for better caching
COPY --chown=$UID:$GID requirements.txt .
RUN --mount=type=cache,uid=$UID,gid=$GID,target=/home/user/.cache/uv \
       uv pip install -r requirements.txt \
       # NOTE: outdated `datasets` gets installed for some reason, so forcefully
       # upgrading it
    && uv pip install --upgrade datasets \
    && :

# Copy the rest of the application
COPY --chown=$UID:$GID . .

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command - you can override this
CMD ["/bin/bash"]
