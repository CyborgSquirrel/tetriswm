#!/usr/bin/env python3
"""
Simple test script to verify GUI functionality works in Docker.
"""
import matplotlib

matplotlib.use('TkAgg')  # Use Tk backend which works well with both X11 and Wayland
import matplotlib.pyplot as plt
import numpy as np


def main():
    """Test basic matplotlib plotting"""
    print("Testing matplotlib GUI...")

    # Create a simple plot
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label='sin(x)')
    plt.plot(x, np.cos(x), label='cos(x)')
    plt.title('GUI Test - Matplotlib in Docker')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

    print("If you see a window with a plot, GUI is working!")
    print("Close the window to exit.")
    plt.show()

if __name__ == '__main__':
    main()
