FROM nvidia/cuda:13.1.1-cudnn-runtime-ubuntu24.04

ARG XUID
ARG XGID

RUN test -n "$XUID" || (echo "XUID build arg is required" && exit 1)
RUN test -n "$XGID" || (echo "XGID build arg is required" && exit 1)

# Set working directory
WORKDIR /app

# Install system dependencies
RUN --mount=type=cache,dst=/var/cache/apt,sharing=locked \
    --mount=type=cache,dst=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    # Misc
        build-essential \
        git \
        swig \
        curl \
        neovim \
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
        qtwayland5 \
        libqt5waylandclient5

# Free up UID 1000
RUN userdel -r ubuntu 2>/dev/null || true

# Create user matching host group and user
RUN \
       groupadd -f -g "$XGID" user \
    && useradd -m -u "$XUID" -g "$XGID" -s /bin/bash user

USER $XUID:$XGID
ENV HOME="/home/user"
ENV PATH="/home/user/.local/bin:$PATH"
RUN \
     mkdir -p /home/user/.cache \
  && mkdir -p /home/user/.local

# Install uv
RUN --mount=type=bind,src=deps/uv/install.sh,dst=/bin/install_uv.sh /bin/install_uv.sh
ENV UV_PYTHON_CACHE_DIR=/home/user/.cache/uv

# Create virtual environment
RUN uv venv /home/user/venv

# Activate venv by modifying PATH
ENV PATH="/home/user/venv/bin:$PATH"

# Copy requirements first for better caching
COPY --chown=$XUID:$XGID requirements.txt .
RUN --mount=type=cache,uid=$XUID,gid=$XGID,dst=/home/user/.cache/uv \
       uv pip install -r requirements.txt \
       # NOTE: outdated `datasets` gets installed for some reason, so forcefully
       # upgrading it
    && uv pip install --upgrade datasets \
    && :

# Copy the rest of the application
COPY --chown=$XUID:$XGID . .

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command - you can override this
CMD ["/bin/bash"]
