# Learning to Play Tetris via World Models

## Development Setup

### Using Docker (Recommended)

**Local development:**
```bash
# Build and start the container
docker-compose up -d

# Enter the container
docker-compose exec app /bin/bash

# Or run a specific script
docker-compose exec app python your_script.py
```

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
