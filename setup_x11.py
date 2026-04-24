#!/usr/bin/env python3
"""Get the X11 authority file path for docker-compose."""

import os
import re
import subprocess
import sys


def main():
    """Extract Xauthority file path from xauth info."""
    try:
        result = subprocess.run(
            ["xauth", "info"],
            capture_output=True,
            text=True,
            check=True
        )

    except subprocess.CalledProcessError as e:
        print(f"Error running xauth: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'xauth' command not found", file=sys.stderr)
        print("Please install xauth on your system", file=sys.stderr)
        sys.exit(1)

    # Parse the "Authority file:" line
    match = re.search(r'^Authority file:\s+(.+)$', result.stdout, re.MULTILINE)

    if not match:
        print("Error: Could not find Authority file in xauth info output", file=sys.stderr)
        sys.exit(1)

    auth_file = match.group(1).strip()
    os.symlink(auth_file, ".Xauthority.docker")

if __name__ == "__main__":
    main()
