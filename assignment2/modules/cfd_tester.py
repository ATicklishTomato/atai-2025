import torch
import matplotlib.pyplot as plt
import wandb
from matplotlib.animation import FuncAnimation
import logging
logger = logging.getLogger(__name__)

def magnitude(tensor):
    # calculates the radial component/magnitude of the 2D velocity field
    return torch.sqrt(tensor[0,:,:]**2 + tensor[1,:,:]**2)

@torch.no_grad()
def generate_single_prediction(
    model, val_dataloader, euler_steps=20, device="cuda",
    save_path="./models/output/prediction.gif", save=False, sigma=0.025,
    condition_on_history=True
):
    logger.info(f"Showing prediction for {euler_steps} euler steps")
    model.to(device)
    model.eval()

    data = next(iter(val_dataloader))
    history_mask, history_sequence, target_mask, target_sequence = data
    history_mask, history_sequence = history_mask.to(device), history_sequence.to(device)
    target_mask, target_sequence = target_mask.to(device), target_sequence.to(device)
    history = torch.cat([history_mask, history_sequence], dim=1)  # [B, C+1, F, W, H]
    target = torch.cat([target_mask, target_sequence], dim=1)  # [B, C+1, F, W, H]

    if condition_on_history:
        x = history + sigma * torch.randn_like(history).to(device)
        noise_padding = torch.randn_like(target).to(device)
        padding_needed = noise_padding.shape[2] - x.shape[2]
        if padding_needed > 0:
            # Prepend random noise to x_0 to match target's frame dimension
            x = torch.cat([noise_padding[:, :, :padding_needed], x], dim=2)
        elif padding_needed < 0:
            x = x[:, :, -padding_needed:]  # Trim x_0 to match target's frame dimension
    else:
        x = torch.randn_like(target).to(device)

    output = model.generation(x, history, euler_steps)  # [B, C+1, F, W, H]
    output = output.cpu().numpy()
    target = target.cpu().numpy()

    _, C, F, W, H = output.shape
    logger.info(f"Show prediction of first batch with channels {C}, frames {F}, width {W}, height {H}")
    fig, axes = plt.subplots(2, 2, figsize=(15, 5))
    for ax in axes.flatten():
        ax.axis("off")

    names = ["Velocity", "Pressure"]

    # Initialize one imshow per subplot (first frame)
    mag_target = magnitude(torch.tensor(target[0, 1:3, 0])).numpy()
    mag_output = magnitude(torch.tensor(output[0, 1:3, 0])).numpy()
    # Compute the vmin and vmax of the whole sequence for consistent color scaling
    whole_mag_target = magnitude(torch.tensor(target[0, 1:3, :])).numpy()
    whole_mag_output = magnitude(torch.tensor(output[0, 1:3, :])).numpy()
    vmin_vel = min(whole_mag_target.min(), whole_mag_output.min())
    vmax_vel = max(whole_mag_target.max(), whole_mag_output.max())
    vmin_press = min(target[0, 3, :].min(), output[0, 3, :].min())
    vmax_press = max(target[0, 3, :].max(), output[0, 3, :].max())
    im_target_vel = axes[0, 0].imshow(mag_target, cmap='viridis', animated=True, vmin=vmin_vel, vmax=vmax_vel)
    im_output_vel = axes[1, 0].imshow(mag_output, cmap='viridis', animated=True, vmin=vmin_vel, vmax=vmax_vel)
    im_target_press = axes[0, 1].imshow(target[0, 3, 0], cmap='viridis', animated=True, vmin=vmin_press, vmax=vmax_press)
    im_output_press = axes[1, 1].imshow(output[0, 3, 0], cmap='viridis', animated=True, vmin=vmin_press, vmax=vmax_press)

    for i, name in enumerate(names):
        axes[0, i].set_title(f"Target {name}")
        axes[1, i].set_title(f"Output {name}")

    def init():
        # You can just return the artists
        return [im_target_vel, im_output_vel, im_target_press, im_output_press]

    def update(frame):
        mag_target = magnitude(torch.tensor(target[0, 1:3, frame])).numpy()
        mag_output = magnitude(torch.tensor(output[0, 1:3, frame])).numpy()

        im_target_vel.set_array(mag_target)
        im_output_vel.set_array(mag_output)
        im_target_press.set_array(target[0, 3, frame])
        im_output_press.set_array(output[0, 3, frame])

        return [im_target_vel, im_output_vel, im_target_press, im_output_press]

    ani = FuncAnimation(fig, update, frames=F, init_func=init, blit=False, interval=200)
    if save:
        ani.save(save_path, writer='ffmpeg')
        logger.info(f"Saved prediction animation to {save_path}")
        wandb.save(save_path)
        logger.info(f"Uploaded prediction animation to wandb")
    plt.close(fig)

    velocity_pred = torch.tensor(output[0, 1:3, :])  # [C=2, F, W, H]
    pressure_pred = torch.tensor(output[0, 3, :])    # [F, W, H]
    velocity_true = torch.tensor(target[0, 1:3, :])    # [C=2, F, W, H]
    pressure_true = torch.tensor(target[0, 3, :])      # [F, W, H]

    velocity_pred = magnitude(velocity_pred)  # [F, W, H]
    velocity_true = magnitude(velocity_true)  # [F, W, H]

    plot_comparison(velocity_pred, velocity_true, pressure_pred, pressure_true,
                    save_path="./models/output/comparison_fields.png", save=save)



def plot_comparison(velocity, velocity_test, pressure, pressure_test,
                    save_path="./models/output/comparison_fields.png", save=False):
    # this function plots the comparison between predicted and true fields at some timesteps
    fig, ax = plt.subplots(4, 4, figsize=(10, 7))
    timesteps = [5, 10, 15, 19]
    for i, t in enumerate(timesteps):
        # turn axis off
        for j in range(4):
            # only remove axis ticks
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            for spine in ax[i, j].spines.values():
                spine.set_visible(False)
        # plot the velocity magnitude and pressure at each time step
        ax[i, 0].imshow(velocity[t].numpy(), cmap='viridis')
        ax[i, 2].imshow(pressure[t].numpy(), cmap='viridis')
        ax[i, 1].imshow(velocity_test[t].numpy(), cmap='viridis')
        ax[i, 3].imshow(pressure_test[t].numpy(), cmap='viridis')
    ax[0, 0].set_title('Predicted Velocity Magnitude')
    ax[0, 2].set_title('Predicted Pressure')
    ax[0, 1].set_title('True Velocity Magnitude')
    ax[0, 3].set_title('True Pressure')
    ax[0, 0].set_ylabel('t={t}'.format(t=timesteps[0]))
    ax[1, 0].set_ylabel('t={t}'.format(t=timesteps[1]))
    ax[2, 0].set_ylabel('t={t}'.format(t=timesteps[2]))
    ax[3, 0].set_ylabel('t={t}'.format(t=timesteps[3]))
    plt.tight_layout()

    if save:
        plt.savefig(save_path)
        logger.info(f"Saved comparison figure to {save_path}")
        wandb.save(save_path)
        logger.info(f"Uploaded comparison figure to wandb")
    plt.close(fig)