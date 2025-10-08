import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display
import logging

from torch import nn

logger = logging.getLogger(__name__)

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
    target = torch.randn_like(target).to(device)

    output = model.generation(target, history, euler_steps)  # [B, C+1, F, W, H]
    output = output.cpu().numpy()
    target = target.cpu().numpy()

    _, C, F, W, H = output.shape
    logger.info(f"Show prediction of first batch with channels {C}, frames {F}, width {W}, height {H}")
    fig, axes = plt.subplots(2, C-1, figsize=(4*(C-1), 8))
    ims = []
    for c in range(1, C):
        # velocity_x, velocity_y, pressure
        names = ["Velocity X", "Velocity Y", "Pressure"]
        ims.append([])
        vmax = np.max(output[:, c])
        for f in range(F):
            im1 = axes[0, c-1].imshow(target[0, c, f], vmin=0, vmax=vmax, cmap='viridis', animated=True)
            im2 = axes[1, c-1].imshow(output[0, c, f], vmin=0, vmax=vmax, cmap='viridis', animated=True)
            if f == 0:
                axes[0, c-1].set_title(f"Target {names[c-1]}")
                axes[1, c-1].set_title(f"Predicted {names[c-1]}")
            ims[c-1].append([im1, im2])

    def init():
        for ax in axes.flatten():
            ax.axis('off')
        return []

    def update(frame):
        for c in range(C-1):
            ims[c][frame][0].set_array(target[0, c+1, frame])
            ims[c][frame][1].set_array(output[0, c+1, frame])
        return [im for sublist in ims for im in sublist[frame]]
    ani = FuncAnimation(fig, update, frames=F, init_func=init, blit=True, interval=200)
    if show:
        plt.show()
    if save:
        ani.save(save_path, writer='pillow')
        logger.info(f"Saved prediction animation to {save_path}")
    plt.close(fig)
