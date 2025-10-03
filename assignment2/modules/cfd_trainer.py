import torch
import wandb
import logging

from torch import nn
from tqdm import tqdm

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
        self.sigma = args.sigma  # noise level
        self.use_tqdm = args.use_tqdm
        self.save = args.save


    def train(self):
        # Implement the training loop
        wandb.watch(self.model, log='all')
        logger.info("Model watched by Weights and Biases")

        for epoch in range(self.epochs):
            self.model.train()
            train_losses = torch.zeros(len(self.train_dataloader))
            if self.use_tqdm:
                loader = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), leave=False, desc=f"Training epoch {epoch+1}/{self.epochs}")
            else:
                loader = enumerate(self.train_dataloader)
            for index, batch in loader:
                batch = batch.to(self.device)
                x_1 = batch # (batch, C, bundle, H, W)
                x_0 = torch.randn_like(x_1).to(self.device)
                t = torch.rand(len(x_1), 1).to(self.device)  # (B,)
                t = t.view(-1, 1, 1, 1, 1)  # (B, 1, 1, 1, 1)
                x_t = (1 - t) * x_0 + t * x_1 + torch.randn_like(x_0) * self.sigma
                u = x_1 - x_0 # (B, C, B, H, W)

                self.optimizer.zero_grad()
                logger.debug(f"Computing training loss for batch {index} with x_t shape: {x_t.shape} and u shape: {u.shape}")
                loss = self.loss_fn(self.model(t=t, x=x_t), u)
                loss.backward()
                train_losses[index] = loss.item()
                self.optimizer.step()

            # Validation step
            self.model.eval()
            with torch.no_grad():
                val_losses = torch.zeros(len(self.val_dataloader))
                if self.use_tqdm:
                    loader = tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader), leave=False, desc=f"Validation epoch {epoch+1}/{self.epochs}")
                else:
                    loader = enumerate(self.val_dataloader)
                for index, batch in loader:
                    batch = batch.to(self.device)
                    x_1 = batch  # (batch, C, bundle, H, W)
                    x_0 = torch.randn_like(x_1).to(self.device)
                    t = torch.rand(len(x_1), 1).to(self.device)  # (B,)
                    t = t.view(-1, 1, 1, 1, 1)  # (B, 1, 1, 1, 1)
                    x_t = (1 - t) * x_0 + t * x_1 + torch.randn_like(x_0) * self.sigma
                    u = x_1 - x_0  # (B, C, B, H, W)

                    logger.debug(f"Computing validation loss for batch {index} with x_t shape: {x_t.shape} and u shape: {u.shape}")
                    loss = self.loss_fn(self.model(t=t, x=x_t), u)
                    val_losses[index] = loss.item()

            wandb.log({"train_avg_loss": torch.mean(train_losses),
                       "val_avg_loss": torch.mean(val_losses)})


            if epoch % (self.epochs // 10) == 0 and epoch > 0:
                if self.save:
                    torch.save(self.model.state_dict(), f"./models/cfd_model_epoch{epoch}.pth")
                    logger.info(f"Model checkpoint saved to ./models/cfd_model_epoch{epoch}.pth")
                    wandb.save(f"./models/cfd_model_epoch{epoch}.pth")
                    logger.info(f"Model checkpoint saved to Weights and Biases for epoch {epoch}")

            logger.info(f"Epoch {epoch+1}/{self.epochs} completed. Train Loss: {torch.mean(val_losses):.6f}, Val Loss: {torch.mean(val_losses):.6f}")

        logger.info("Training completed.")
        if self.save:
            torch.save(self.model.state_dict(), "./models/cfd_model.pth")
            logger.info("Model saved to ./models/cfd_model.pth")
            wandb.save("./models/cfd_model.pth")
            logger.info("Model checkpoint saved to Weights and Biases")