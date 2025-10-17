from os import makedirs
from os.path import dirname

import torch 
from torch import Tensor

import matplotlib.pyplot as plt
from matplotlib import animation


def square_torus_distance(p1, p2, width=1.0, height=1.0):
    """
    Compute the squared torus distance between two points p1 and p2 in a box of given width and height.

    Args:
        p1: Tensor of shape (num_points, 2) representing the first point(s)
        p2: Tensor of shape (num_points, 2) representing the second point(s)
        width: Width of the box
        height: Height of the box
    Returns:
        dist: Tensor of shape (num_points,) representing the squared torus distance between p1 and p2
    """
    dx = p1[:, 0] - p2[:, 0]
    dy = p1[:, 1] - p2[:, 1]
    dx = dx - width * torch.round(dx / width)
    dy = dy - height * torch.round(dy / height)
    return dx ** 2 + dy ** 2

def pbc_direction(p1: Tensor, p2: Tensor) -> Tensor:
    """
    Compute the direction from p1 to p2 considering periodic boundary conditions in a unit square.
    Args:
        p1: Tensor of shape (num_points, 2) representing the first point(s)
        p2: Tensor of shape (num_points, 2) representing the second point(s)
    Returns:
        direction: Tensor of shape (num_points, 2) representing the direction from p1 to p2 considering PBCs
    """
    return p2 - p1 - torch.round(p2 - p1)

def animate_rollout(rollouts, output_path="output/rollout.gif", width = 1.0, height = 1.0, max_timesteps=100):
    # rollouts of shape (Timesteps, Boids, Node_dim)
    
    # Create output directory if it does not exist
    makedirs(dirname(output_path), exist_ok=True)

    # Parse rollouts
    timesteps, num_boids, node_dim = rollouts.shape
    rollouts = rollouts.cpu().numpy()

    # Initialize the figure and axis
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)

    quiv = ax.quiver(rollouts[0, :, 0], rollouts[0, :, 1], rollouts[0, :, 2], rollouts[0, :, 3])
    scat = ax.scatter(rollouts[0, :, 0], rollouts[0, :, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal', adjustable='box')

    def update(frame):
        quiv.set_offsets(rollouts[frame, :, :2])
        quiv.set_UVC(rollouts[frame, :, 2], rollouts[frame, :, 3])
        scat.set_offsets(rollouts[frame, :, :2])
        return scat, quiv

    ani = animation.FuncAnimation(fig, update, frames=min(timesteps, max_timesteps), blit=False, interval=150)
    ani.save(output_path, writer="ffmpeg")
    plt.close()
