import torch
import wandb
import logging

from torch import nn
from tqdm import tqdm

logger = logging.getLogger(__name__)

class CFDTrainer():
    def __init__(self, args, model, train_dataloader, val_dataloader):
        self.epochs = args.epochs
        self.device = args.device
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.sigma = args.sigma  # noise level
        self.use_tqdm = args.use_tqdm
        self.save = args.save
        self.patience = args.patience
        self.condition_on_history = args.prior_conditioning


    def train(self):
        # Implement the training loop
        wandb.watch(self.model, log='all')
        logger.info("Model watched by Weights and Biases")
        best_val_loss = float('inf')
        patience = self.patience

        for epoch in range(self.epochs):
            self.model.train()
            train_losses = torch.zeros(len(self.train_dataloader))
            if self.use_tqdm:
                loader = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), leave=False, desc=f"Training epoch {epoch+1}/{self.epochs}")
            else:
                loader = enumerate(self.train_dataloader)
            for index, data in loader:
                history_mask, history_sequence, target_mask, target_sequence = data
                history_mask, history_sequence = history_mask.to(self.device), history_sequence.to(self.device)
                target_mask, target_sequence = target_mask.to(self.device), target_sequence.to(self.device)
                logger.debug(f"History mask shape: {history_mask.shape}, history sequence shape: {history_sequence.shape}, " +
                                f"target mask shape: {target_mask.shape}, target sequence shape: {target_sequence.shape}")
                history = torch.cat([history_mask, history_sequence], dim=1)  # [B, C+1, F, W, H]
                target = torch.cat([target_mask, target_sequence], dim=1)  # [B, C+1, F, W, H]

                if self.condition_on_history:
                    x_0 = history + self.sigma * torch.randn_like(history).to(self.device)
                    logger.debug(f"x_0 shape before conditioning: {x_0.shape}")
                    noise_padding = torch.randn_like(target).to(self.device)
                    padding_needed = noise_padding.shape[2] - x_0.shape[2]
                    logger.debug(f"padding needed: {padding_needed}")
                    if padding_needed > 0:
                        # Prepend random noise to x_0 to match target's frame dimension
                        x_0 = torch.cat([noise_padding[:, :, :padding_needed], x_0], dim=2)
                    elif padding_needed < 0:
                        x_0 = x_0[:, :, -padding_needed:]  # Trim x_0 to match target's frame dimension
                    logger.debug(f"x_0 shape after conditioning: {x_0.shape}")
                else:
                    x_0 = torch.randn_like(target).to(self.device)
                t = torch.rand(target.shape[0], 1,).to(self.device)  # (B,)
                t = t.view(-1, 1, 1, 1, 1)  # (B,1,1,1,1)
                x_t = (1 - t) * x_0 + t * target + torch.randn_like(x_0) * self.sigma


                true_vel = target - x_0 # (B, C+1, F, W, H)
                pred_vel = self.model(t=t, x_t=x_t, x_init=history)  # (B, C+1, F, W, H)


                self.optimizer.zero_grad()
                logger.debug(f"Computing training loss for batch {index} with x_t shape: {pred_vel.shape} " +
                             f"and u shape: {true_vel.shape}")
                loss = self.loss_fn(pred_vel, true_vel)
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
                for index, data in loader:
                    history_mask, history_sequence, target_mask, target_sequence = data
                    history_mask, history_sequence = history_mask.to(self.device), history_sequence.to(self.device)
                    target_mask, target_sequence = target_mask.to(self.device), target_sequence.to(self.device)
                    history = torch.cat([history_mask, history_sequence], dim=1)  # [B, C+1, F, W, H]
                    target = torch.cat([target_mask, target_sequence], dim=1)  # [B, C+1, F, W, H]

                    if self.condition_on_history:
                        x_0 = history + self.sigma * torch.randn_like(history).to(self.device)
                        logger.debug(f"x_0 shape before conditioning: {x_0.shape}")
                        noise_padding = torch.randn_like(target).to(self.device)
                        padding_needed = noise_padding.shape[2] - x_0.shape[2]
                        logger.debug(f"padding needed: {padding_needed}")
                        if padding_needed > 0:
                            # Prepend random noise to x_0 to match target's frame dimension
                            x_0 = torch.cat([noise_padding[:, :, :padding_needed], x_0], dim=2)
                        elif padding_needed < 0:
                            x_0 = x_0[:, :, -padding_needed:]  # Trim x_0 to match target's frame dimension
                        logger.debug(f"x_0 shape after conditioning: {x_0.shape}")
                    else:
                        x_0 = torch.randn_like(target).to(self.device)
                    t = torch.rand(target.shape[0], 1, ).to(self.device)  # (B,)
                    t = t.view(-1, 1, 1, 1, 1)  # (B,1,1,1,1)

                    x_t = (1 - t) * x_0 + t * target + torch.randn_like(x_0) * self.sigma
                    true_vel = target - x_0  # (B, C+1, F, W, H)
                    pred_vel = self.model(t=t, x_t=x_t, x_init=history)  # (B, C+1, F, W, H)

                    logger.debug(f"Computing validation loss for batch {index} with x_t shape: {pred_vel.shape} " +
                        f"and u shape: {true_vel.shape}")
                    loss = self.loss_fn(pred_vel, true_vel)
                    val_losses[index] = loss.item()

            wandb.log({"train_avg_loss": torch.mean(train_losses),
                       "val_avg_loss": torch.mean(val_losses)})


            if epoch % (self.epochs // 10) == 0 and epoch > 0:
                if self.save:
                    torch.save(self.model.state_dict(), f"./models/cfd_model_epoch{epoch}.pth")
                    logger.info(f"Model checkpoint saved to ./models/cfd_model_epoch{epoch}.pth")
                    wandb.save(f"./models/cfd_model_epoch{epoch}.pth")
                    logger.info(f"Model checkpoint saved to Weights and Biases for epoch {epoch}")

            logger.info(f"Epoch {epoch+1}/{self.epochs} completed. Train Loss: {torch.mean(train_losses):.6f}, Val Loss: {torch.mean(val_losses):.6f}")
            # Early stopping
            if torch.mean(val_losses) < best_val_loss:
                best_val_loss = torch.mean(val_losses)
                patience = self.patience
            else:
                patience -= 1
                logger.info(f"No improvement in validation loss. Patience left: {patience}")
                if patience <= 0:
                    logger.info("Early stopping triggered.")
                    break

        logger.info("Training completed.")
        if self.save:
            torch.save(self.model.state_dict(), "./models/cfd_model.pth")
            logger.info("Model saved to ./models/cfd_model.pth")
            wandb.save("./models/cfd_model.pth")
            logger.info("Model checkpoint saved to Weights and Biases")