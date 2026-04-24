# Learning to Play Tetris via World Models

## Development Setup

### Using Docker (Recommended)

**Local development:**
```bash
# First time setup: Generate X11 authentication for GUI support
./setup_x11.py

# Build and start the container
docker-compose up -d

# Enter the container
docker-compose exec app /bin/bash

# Or run a specific script
docker-compose exec app python your_script.py

# Test GUI functionality
docker-compose exec app python test_gui.py
```

**GUI Support:**
The Docker setup supports GUI applications on Wayland (with X11/XWayland fallback). This allows you to:
- Render Tetris games visually
- Display matplotlib plots
- Use Jupyter notebooks with inline graphics

**Important:** Run `./setup_x11.py` before starting the container to enable GUI support. This script generates X11 authentication credentials that allow the container to display windows on your host.

If you encounter GUI issues:
- Re-run `./setup_x11.py` and restart the container
- Check that `$XDG_RUNTIME_DIR` and `$WAYLAND_DISPLAY` are set on your host
- Verify `$DISPLAY` is set (usually `:0` or `:1`)

**For GPU cluster:**
```bash
# Build GPU-enabled image
docker build -f Dockerfile.gpu -t tetris-worldmodel:gpu .

# Run with GPU support
docker run --gpus all -it tetris-worldmodel:gpu
```

### Using venv (Alternative)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Project Structure

*To be added as the project develops*

## Usage

*To be added*
