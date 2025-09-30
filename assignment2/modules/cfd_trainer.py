import torch
import wandb
import logging

from torch import nn

logger = logging.getLogger(__name__)

class CFDTrainer():
    # TODO: Implement the trainer class for CFD problem
    def __init__(self, args, model, train_dataloader, val_dataloader):
        self.epochs = args.epochs
        self.device = args.device
        self.model = model.to(self.device)
        self.skip_train = args.skip_train
        self.skip_val = args.skip_test
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.sigma = 0.1  # noise level


    def train(self):
        # Implement the training loop
        wandb.watch(self.model, log='all', log_freq=25)
        logger.info("Model watched by Weights and Biases")

        for epoch in range(self.epochs):
            self.model.train()
            losses = torch.zeros(len(self.train_dataloader))
            for index, batch in enumerate(self.train_dataloader):
                mask, _, input, target = batch
                # Forward pass, compute loss, backward pass, optimizer step
                mask, input, target = mask.to(self.device), input.to(self.device), target.to(self.device)
                input = torch.cat([mask, input], dim=1)
                x_1 = target  # (B, 1, 28, 28)
                x_0 = torch.randn_like(x_1).to(self.device)
                t = torch.rand(len(x_1), 1).to(self.device)  # (B,)
                t = t.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
                x_t = (1 - t) * x_0 + t * x_1 + torch.randn_like(x_0) * self.sigma
                u = x_1 - x_0  # (B, 1, 28, 28)

                self.optimizer.zero_grad()
                loss = self.loss_fn(self.model(t=t, x=x_t), u)
                loss.backward()
                losses[index] = loss.item()
                self.optimizer.step()

            wandb.log({'train_total_loss': torch.sum(losses),
                       "train_avg_loss": torch.mean(losses)
                       })

            # Validation step
            self.model.eval()
            with torch.no_grad():
                losses = torch.zeros(len(self.val_dataloader))
                for index, batch in enumerate(self.val_dataloader):
                    # Forward pass and compute validation metrics
                    x_1 = batch.to(self.device)  # (B, 1, 28, 28)
                    x_0 = torch.randn_like(x_1).to(self.device)
                    t = torch.rand(len(x_1), 1).to(self.device)  #
                    t = t.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
                    x_t = (1 - t) * x_0 + t * x_1 + torch.randn_like(x_0) * self.sigma
                    u = x_1 - x_0  # (B, 1, 28, 28)

                    loss = self.loss_fn(self.model(t=t, x=x_t), u)
                    losses[index] = loss.item()

                wandb.log({'val_total_loss': torch.sum(losses),
                           "val_avg_loss": torch.mean(losses)
                           })


            if epoch % (self.epochs // 10) == 0 and epoch > 0:
                torch.save(self.model.state_dict(), f"./models/cfd_model_epoch{epoch}.pth")
                logger.info(f"Model checkpoint saved to ./models/cfd_model_epoch{epoch}.pth")
                wandb.save(f"./models/cfd_model_epoch{epoch}.pth")
                logger.info(f"Model checkpoint saved to Weights and Biases for epoch {epoch}")

            logger.info(f"Epoch {epoch+1}/{self.epochs} completed.")

        logger.info("Training completed.")
        torch.save(self.model.state_dict(), "./models/cfd_model.pth")
        logger.info("Model saved to ./models/cfd_model.pth")
        wandb.save("./models/cfd_model.pth")
        logger.info("Model checkpoint saved to Weights and Biases")