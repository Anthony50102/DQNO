import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animation(images: np.array) -> FuncAnimation:
    fig, ax = plt.subplots()
    im = ax.imshow(images[0], cmap='viridis', animated=True)

    def update(frame):
        im.set_array(images[frame])
        ax.set_title(f"Frame {frame}")
        return [im]

    ani = FuncAnimation(fig, update, frames=len(images), blit=True, interval=50)
    return ani