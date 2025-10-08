import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display
import logging

from torch import nn

logger = logging.getLogger(__name__)

def magnitude(tensor):
    # calculates the radial component/magnitude of the 2D velocity field
    return torch.sqrt(tensor[0,:,:]**2 + tensor[1,:,:]**2)

@torch.no_grad()
def show_prediction(model, val_dataloader, euler_steps=20, device="cuda", show=True,
                    save_path="./models/output/prediction.gif", save=False):
    logger.info(f"Showing prediction for {euler_steps} euler steps")
    model.to(device)
    model.eval()

    data = next(iter(val_dataloader))
    history_mask, history_sequence, target_mask, target_sequence = data
    history_mask, history_sequence = history_mask.to(device), history_sequence.to(device)
    target_mask, target_sequence = target_mask.to(device), target_sequence.to(device)
    history = torch.cat([history_mask, history_sequence], dim=1)  # [B, C+1, F, W, H]
    target = torch.cat([target_mask, target_sequence], dim=1)  # [B, C+1, F, W, H]
    noise = torch.randn_like(target).to(device)

    output = model.generation(noise, history, euler_steps)  # [B, C+1, F, W, H]
    output = output.cpu().numpy()
    target = target.cpu().numpy()

    _, C, F, W, H = output.shape
    logger.info(f"Show prediction of first batch with channels {C}, frames {F}, width {W}, height {H}")
    fig, axes = plt.subplots(2, 2, figsize=(15, 5))
    ims = []
    names = ["Velocity", "Pressure"]
    for frame in range(F):
        # Compute magnitudes for velocity channels
        mag_target = magnitude(torch.tensor(target[0, 1:3, frame])).numpy()
        mag_output = magnitude(torch.tensor(output[0, 1:3, frame])).numpy()
        vmax_vel = max(np.max(np.abs(mag_target)), np.max(np.abs(mag_output)))
        vmin_vel = min(np.min(np.abs(mag_target)), np.min(np.abs(mag_output)))
        # For pressure channel
        vmax_press = max(np.max(np.abs(target[0, 3, frame])), np.max(np.abs(output[0, 3, frame])))
        vmin_press = min(np.min(np.abs(target[0, 3, frame])), np.min(np.abs(output[0, 3, frame])))
        ims_frame = []
        for c in range(len(names)):
            if c == 0:  # Velocity channel
                im_target = axes[0, c].imshow(mag_target, vmin=vmin_vel, vmax=vmax_vel, cmap='viridis', animated=True)
                im_output = axes[1, c].imshow(mag_output, vmin=vmin_vel, vmax=vmax_vel, cmap='viridis', animated=True)
            else:  # Pressure channel
                im_target = axes[0, c].imshow(target[0, c+1, frame], vmin=vmin_press, vmax=vmax_press, cmap='viridis', animated=True)
                im_output = axes[1, c].imshow(output[0, c+1, frame], vmin=vmin_press, vmax=vmax_press, cmap='viridis', animated=True)
            if frame == 0:
                axes[0, c].set_title(f"Target {names[c]}")
                axes[1, c].set_title(f"Output {names[c]}")
            ims_frame.append((im_target, im_output))
        ims.append(ims_frame)



    def init():
        for ax in axes.flatten():
            ax.axis('off')
        return []

    def update(frame):
        ims_frame = ims[frame]
        artists = []
        for c in range(len(names)):
            im_target, im_output = ims_frame[c]
            axes[0, c].images[0].set_array(im_target.get_array())
            axes[1, c].images[0].set_array(im_output.get_array())
            artists.extend([axes[0, c].images[0], axes[1, c].images[0]])
        return artists
    ani = FuncAnimation(fig, update, frames=F, init_func=init, blit=True, interval=200)
    if show:
        plt.show()
    if save:
        ani.save(save_path, writer='pillow')
        logger.info(f"Saved prediction animation to {save_path}")
    plt.close(fig)
