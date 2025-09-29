import torch
import wandb
import logging

logger = logging.getLogger(__name__)

class BoidsTrainer():
    # TODO: Implement the trainer class for boids problem
    def __init__(self, args, model, train_dataloader, val_dataloader):
        self.epochs = args.epochs
        self.device = args.device
        self.model = model.to(self.device)
        self.skip_train = args.skip_train
        self.skip_val = args.skip_test
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        # Initialize other components like optimizer, loss function, etc.

    def train(self):
        # Implement the training loop
        wandb.watch(self.model, log='all', log_freq=250)
        logger.info("Model watched by Weights and Biases")

        for epoch in range(self.epochs):
            self.model.train()
            losses = []
            for batch in self.train_dataloader:
                # Forward pass, compute loss, backward pass, optimizer step
                pass

            wandb.log({'train_total_loss': torch.sum(losses),
                       "train_avg_loss": torch.mean(losses)
                       })

            # Validation step
            self.model.eval()
            with torch.no_grad():
                losses = []
                for batch in self.val_dataloader:
                    # Forward pass and compute validation metrics
                    pass

                wandb.log({'val_total_loss': torch.sum(losses),
                           "val_avg_loss": torch.mean(losses)
                           })