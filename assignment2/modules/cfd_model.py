import torch
from torch import nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch)
        self.norm2 = nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch)
        if in_ch != out_ch:
            self.nin = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        else:
            self.nin = nn.Identity()

    def forward(self, x):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.relu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        out = F.relu(h + self.nin(x))
        return out


# -------------------------
# UNet-like architecture (with t as extra channel)
# -------------------------
class CFDModel(nn.Module):
    def __init__(self, in_ch=4, base_ch=64): # in_ch=4 for CFD (mask+vx+vy+p)
        super().__init__()
        # encoder
        self.enc1 = ResBlock(in_ch + 1, base_ch)  # +1 for FM's time channel
        self.enc2 = ResBlock(base_ch, base_ch * 2)
        self.enc3 = ResBlock(base_ch * 2, base_ch * 4)
        # decoder
        self.dec3 = ResBlock(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.dec2 = ResBlock(base_ch * 2 + base_ch, base_ch)
        self.out_conv = nn.Sequential(
            nn.Conv3d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(base_ch, in_ch, kernel_size=1)
        )
        self.pool = nn.AvgPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, t, x):
        # x: (batch, B, C, H, W), t: (batch, B, 1, 1, 1) in [0,1]
        logger.debug(f"Forward pass with x shape: {x.shape} and t shape: {t.shape}")
        batch_size, C, B, H, W = x.shape
        # expand t to (batch, B,1,H,W)
        t_map = t.expand(batch_size, 1, B, H, W)
        xt = torch.cat([x, t_map], dim=1)  # concat along channel
        # encoder
        e1 = self.enc1(xt)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        # decoder
        d3 = self.upsample(e3)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        d2 = self.upsample(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        out = self.out_conv(d2)
        logger.debug(f"Forward pass with out shape: {out.shape}")
        return out

    def generation(self, x, n_euler_steps, t_start=0.0, t_end=1.0):
        logger.debug(f"Generation with x shape: {x.shape}, n_euler_steps: {n_euler_steps}, t_start: {t_start}, t_end: {t_end}")
        time_steps = torch.linspace(t_start, t_end, n_euler_steps + 1).view(-1, 1, 1, 1, 1).to(x.device)

        for i in range(n_euler_steps):
            x = x + (time_steps[i + 1] - time_steps[i]) * self(t=time_steps[i], x=x)

        logger.debug(f"Generation with x shape: {x.shape}")
        return x
