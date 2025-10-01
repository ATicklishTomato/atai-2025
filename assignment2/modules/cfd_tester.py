import math
import wandb
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import logging

from torch import nn

logger = logging.getLogger(__name__)

def compute_rollout(model, num_steps, bundle_size, device, initial_state=None, true_trajectory=None):
    logger.info("Starting rollout computation")
    model.eval()
    with torch.no_grad():
        if initial_state is None:
            # start from random noise if no initial state is provided
            current_state = torch.randn(1, 4, bundle_size, 64, 128).to(device)  # [1, C, B, H, W]
        else:
            current_state = initial_state.to(device)  # [1, C, B, H, W]

        trajectory = [current_state.cpu().numpy()]

        for step in range(math.ceil(num_steps / bundle_size)):
            next_state = model.generation(current_state, n_euler_steps=10, t_start=0.0, t_end=1.0)
            trajectory.append(next_state.cpu().numpy())
            current_state = next_state
            if true_trajectory is not None and step < math.ceil(num_steps / bundle_size) - 1:
                # We predict bundle_size frames at a time, so we can compare with true trajectory if available
                true_next = true_trajectory[:, :, step * bundle_size:(step + 1) * bundle_size, :, :].to(device)
                loss = nn.MSELoss()(trajectory, true_next)
                wandb.log({"Test loss": loss.item()})

    trajectory = np.concatenate(trajectory, axis=2)  # concatenate along time dimension

    # reshape to [num_steps+1, C, H, W]
    trajectory = trajectory[0]  # remove batch dimension
    trajectory = trajectory.transpose(1, 0, 2, 3)  # [C, T, H, W]
    trajectory = trajectory[:, :num_steps + 1, :, :]  # ensure correct length
    trajectory = trajectory.transpose(1, 0, 2, 3)  # [T, C, H, W]
    logger.info("Rollout computation completed")
    return trajectory

def animate_rollout(trajectory, interval=50, save_path="../models/output/cfd_rollout.gif", show=False, save=True):
    logger.info(f"Starting animation, saving to {save_path}")
    dir = os.path.dirname(save_path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    num_steps, C, H, W = trajectory.shape

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ims = []

    for t in range(num_steps):
        im1 = ax[0].imshow(trajectory[t, 1], vmin=-1, vmax=1, animated=True)  # vx
        im2 = ax[1].imshow(trajectory[t, 2], vmin=-1, vmax=1, animated=True)  # vy
        im3 = ax[2].imshow(trajectory[t, 3], vmin=-1, vmax=1, animated=True)  # p
        if t == 0:
            ax[0].set_title('Velocity X')
            ax[1].set_title('Velocity Y')
            ax[2].set_title('Pressure')
        ims.append([im1, im2, im3])

    ani = FuncAnimation(fig, lambda i: ims[i], frames=num_steps, interval=interval, blit=True)
    if save:
        ani.save(save_path, writer='pillow')
        wandb.save(save_path)
    if show:
        plt.show()
    logger.info(f"Saved animation to {save_path}")
