import torch
import wandb
import logging

from tqdm import tqdm

from boids_helper_functions import pbc_direction
from boids_dataloaders import BoidsDataset
from boids_model import BoidsModel

logger = logging.getLogger(__name__)


def boids_path_sampler(t, x_1, x_c, sigma):
    # Initialize x_t and x_dot_t
    x_t = torch.zeros_like(x_1)
    x_dot_t = torch.zeros_like(x_1)
    
    # Sample standard normal noise
    noise = torch.randn_like(x_1)
    sigma_x, sigma_v = sigma

    # Compute x_t and x_dot_t
    x_t[:, :2] = (x_c[:, :2] + pbc_direction(x_c[:, :2], x_1[:, :2]) * t + sigma_x * noise[:, :2]) % 1.0
    x_t[:, 2:] = x_c[:, 2:] + (x_1[:, 2:] - x_c[:, 2:]) * t + sigma_v * noise[:, 2:]
    x_dot_t[:, :2] = pbc_direction(x_c[:, :2], x_1[:, :2])
    x_dot_t[:, 2:] = (x_1[:, 2:] - x_c[:, 2:])
    
    return x_t, x_dot_t

def custom_loss(a, b):
    return torch.nn.functional.mse_loss(a[:, :2], b[:, :2]) + torch.nn.functional.mse_loss(100 * a[:, 2:], 100 * b[:, 2:])

class BoidsTrainer():
    def __init__(self, args, model, train_dataloader, val_dataloader):
        self.epochs: int = args.epochs
        self.device: str = args.device
        self.VF: BoidsModel = model.to(self.device)
        self.skip_train: bool = args.skip_train
        self.skip_val: bool = args.skip_test
        self.train_dataset: BoidsDataset = train_dataloader
        self.val_dataset: BoidsDataset = val_dataloader

        self.sigma: tuple[float, float] = (args.sigma, args.sigma / 100)  # noise level TODO: check whether this can be a tuple
        
        # Initialize other components like optimizer, loss function, etc.
        self.loss_fn = custom_loss
        self.optimizer = torch.optim.Adam(self.VF.parameters(), lr=args.lr)
        self.use_tqdm = args.use_tqdm
        self.save = args.save

        self.model_name = "boids_model"

    def _tqdmify(self, iterator, epoch, validation=False):
        if validation:
            desc = "Validation epoch"
        else:
            desc = "Epoch"

        if self.use_tqdm:
            return tqdm(iterator, total=len(iterator), leave=False, desc=f"{desc} {epoch+1}/{self.epochs}")
        else:
            return iterator

    def train(self):
        # Implement the training loop
        wandb.watch(self.model, log='all')
        logger.info("Model watched by Weights and Biases")

        n_train_samples = len(self.train_dataset)

        for epoch in range(self.epochs):
            # Training step
            self.VF.train()
            train_losses = []

            # Go through the training dataset in a random order
            permuted_idxs = torch.randperm(n_train_samples)
            for idx in self._tqdmify(permuted_idxs, epoch):
                data = self.train_dataset[idx]

                data_c = data.clone().to(self.device)
                data_t = data.clone().to(self.device)

                x_c = data_t.x
                x_1 = data_t.y
                t = torch.rand(1).to(self.device)

                x_t, u = boids_path_sampler(t, x_1, x_c, self.sigma)

                data_t.x = x_t
                
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.VF(t=t, data_t=data_t, data_c=data_c), u)
                loss.backward()
                train_losses.append(loss.item())
                self.optimizer.step()

            # Validation step
            self.VF.eval()
            with torch.no_grad():
                val_losses = []

                for data in self._tqdmify(self.val_dataset, epoch, validation=True):
                    # Forward pass and compute validation metrics
                    data_c = data.clone().to(self.device)
                    data_t = data.clone().to(self.device)

                    x_c = data_t.x
                    x_1 = data_t.y
                    t = torch.rand(1).to(self.device)

                    x_t, u = boids_path_sampler(t, x_1, x_c, self.sigma)

                    data_t.x = x_t
                    
                    loss = self.loss_fn(self.VF(t=t, data_t=data_t, data_c=data_c), u)
                    val_losses.append(loss.item())
            
            # Log results
            wandb.log({"train_avg_loss": torch.mean(train_losses), "val_avg_loss": torch.mean(val_losses)})

        # Save the final model
        logger.info("Training completed.")
        if self.save:
            torch.save(self.VF.state_dict(), f"./models/{self.model_name}.pth")
            logger.info(f"Model saved to ./models/{self.model_name}.pth")
            wandb.save(f"./models/{self.model_name}.pth")
            logger.info("Model checkpoint saved to Weights and Biases")

