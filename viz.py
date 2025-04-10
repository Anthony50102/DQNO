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

def animation_comparison(ground_truth: np.array, prediction: np.array, show_difference: bool = True) -> FuncAnimation:
    """
    Create an animation comparing ground truth vs prediction side by side.
    
    Parameters:
    -----------
    ground_truth : np.array
        Array of ground truth images with shape (frames, height, width) or (frames, height, width, channels)
    prediction : np.array
        Array of prediction images with same shape as ground truth
    show_difference : bool, optional
        Whether to show the difference between ground truth and prediction (default=True)
    
    Returns:
    --------
    FuncAnimation
        Animation object that can be displayed or saved
    """
    # Verify inputs have the same shape and length
    if ground_truth.shape != prediction.shape:
        raise ValueError(f"Ground truth and prediction must have the same shape. Got {ground_truth.shape} and {prediction.shape}")
    
    num_frames = ground_truth.shape[0]
    
    # Determine number of subplots (2 or 3 depending on show_difference)
    n_cols = 3 if show_difference else 2
    
    # Create figure and axes
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
    fig.tight_layout(pad=5.0)
    
    # Initialize the plots with first frame
    imgs = []
    imgs.append(axes[0].imshow(ground_truth[0], cmap='viridis', animated=True))
    axes[0].set_title("Ground Truth")
    axes[0].axis('off')
    
    imgs.append(axes[1].imshow(prediction[0], cmap='viridis', animated=True))
    axes[1].set_title("Prediction")
    axes[1].axis('off')
    
    if show_difference:
        # For difference plot, use a different colormap to highlight differences
        difference = ground_truth[0] - prediction[0]
        vmax = max(abs(np.min(difference)), abs(np.max(difference)))
        imgs.append(axes[2].imshow(difference, cmap='bwr', animated=True, vmin=-vmax, vmax=vmax))
        axes[2].set_title("Difference (GT - Pred)")
        axes[2].axis('off')
    
    # Add a main title with frame counter
    main_title = fig.suptitle(f"Frame: 0/{num_frames-1}", fontsize=16)
    
    def update(frame):
        imgs[0].set_array(ground_truth[frame])
        imgs[1].set_array(prediction[frame])
        
        if show_difference:
            difference = ground_truth[frame] - prediction[frame]
            # Recalculate vmax for balanced difference visualization
            vmax = max(abs(np.min(difference)), abs(np.max(difference)))
            imgs[2].set_array(difference)
            imgs[2].set_clim(-vmax, vmax)
        
        main_title.set_text(f"Frame: {frame}/{num_frames-1}")
        return imgs
    
    ani = FuncAnimation(fig, update, frames=num_frames, blit=True, interval=100)
    return ani