# syntax=docker/dockerfile:1.4

FROM nvidia/cuda:13.1.1-cudnn-runtime-ubuntu24.04

SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

ARG XUID
ARG XGID

RUN <<EOF
test -n "$XUID" || (echo "XUID build arg is required" && exit 1)
test -n "$XGID" || (echo "XGID build arg is required" && exit 1)
EOF

# Set working directory
WORKDIR /app

# Install system dependencies
RUN --mount=type=cache,dst=/var/cache/apt,sharing=locked \
    --mount=type=cache,dst=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    # Misc
        build-essential \
        python3-dev \
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

RUN <<EOF
# Free up UID 1000
userdel -r ubuntu 2>/dev/null || true

# Create user matching host group and user
groupadd -f -g "$XGID" user
useradd -m -u "$XUID" -g "$XGID" -s /bin/bash user
chown user:user /app
EOF

USER $XUID:$XGID
ENV HOME="/home/user"
ENV PATH="/home/user/.local/bin:$PATH"
RUN <<EOF
mkdir -p /home/user/.cache
mkdir -p /home/user/.local
EOF

# Install uv
RUN --mount=type=bind,src=deps/uv/install.sh,dst=/bin/install_uv.sh /bin/install_uv.sh
ENV UV_PYTHON_CACHE_DIR=/home/user/.cache/uv

# Create virtual environment
RUN uv venv /home/user/venv

# Activate venv by modifying PATH
ENV UV_PROJECT_ENVIRONMENT=/home/user/venv
ENV VIRTUAL_ENV=/home/user/venv
ENV PATH="/home/user/venv/bin:$PATH"

# NOTE(andrei): It has been roughly a month since we last generated the
# requirements.lock file, and if we try to regenerate it now, it resolves to
# dependencies which cause breakage in the project.
#
# So for the time being we'll use the lockfile instead of requirements.txt.

# Copy requirements first for better caching
# COPY --chown=$XUID:$XGID requirements.txt .
COPY --chown=$XUID:$XGID requirements.lock .
RUN --mount=type=cache,uid=$XUID,gid=$XGID,dst=/home/user/.cache/uv \
<<EOF
# uv pip compile requirements.txt -o requirements.lock
uv pip sync requirements.lock
# NOTE: outdated `datasets` gets installed for some reason, so forcefully
# upgrading it
uv pip install --upgrade datasets
uv pip install 'git+https://github.com/CyborgSquirrel/stable-worldmodel.git@patched'
EOF

# Copy the rest of the application
COPY --chown=$XUID:$XGID . .

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command - you can override this
CMD ["/bin/bash"]
