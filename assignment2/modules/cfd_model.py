import torch
from torch import nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class HistoryEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.GELU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(hidden_channels, 1, kernel_size=1)  # compress to single channel
        )

    def forward(self, history):
        """
        Args:
            history: (B, C, F, W, H)
        Returns:
            encoded_history: (B, 1, F, W, H)
        """
        return self.net(history)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1, do_dropout=False):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch)
        self.norm2 = nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.do_dropout = do_dropout
        if in_ch != out_ch:
            self.nin = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        else:
            self.nin = nn.Identity()

    def forward(self, x):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.gelu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        out = F.gelu(h + self.nin(x))
        if self.do_dropout:
            out = self.dropout(out)
        return out


# -------------------------
# UNet-like architecture (with t as extra channel)
# -------------------------
class CFDModel(nn.Module):
    def __init__(self, in_ch=4, base_ch=64, ch_mults=[1, 2, 2], num_layers=3, dropout=0.1):
        # in_ch=4 for CFD (mask+vx+vy+p) + 1 for FM time
        super().__init__()

        assert num_layers == len(ch_mults), "num_layers must match length of ch_mults"

        self.history_encoder = HistoryEncoder(in_channels=in_ch, hidden_channels=base_ch, dropout=0.1)

        self.enc1 = ResBlock(in_ch + 1, base_ch, dropout=dropout, do_dropout=True)  # +1 for time channel
        self.next_encoders = nn.ModuleList(
            [ResBlock(base_ch * ch_mults[i], base_ch * ch_mults[i + 1], dropout=dropout, do_dropout=True) for i in range(len(ch_mults) - 1)]
        )

        self.avg_pool = nn.AvgPool3d(kernel_size=2, stride=2)
        self.interpolate = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.init_dec = nn.ModuleList(
            [ResBlock(base_ch * ch_mults[i + 1] + base_ch * ch_mults[i]  # cat with skip connection
                      , base_ch * ch_mults[i], dropout=dropout, do_dropout=True) for i in reversed(range(len(ch_mults) - 1))]
        )
        self.out_dec = ResBlock(base_ch * ch_mults[0], in_ch, do_dropout=False) # No dropout at end like CFD practical


    def forward(self, t, x_t, x_init):
        # t: (B, 1, 1, 1, 1), x_t: (B, C, F, W, H), x_init: (B, C, F, W, H)
        logger.debug(f"Forward pass with t shape: {t.shape}, x_t shape: {x_t.shape}, x_init shape: {x_init.shape}")
        # x = torch.cat([x_init, x_t], dim=2)  # (B, C, F, W, H)
        x = x_t
        B, C, F, W, H = x.shape
        t_channel = t.expand(B, 1, F, W, H)  # (B, 1, F, W, H)
        x = torch.cat([x, t_channel], dim=1)  # (B, C+1, F, W, H)
        logger.debug(f"Forward pass with shape {x.shape}")
        h = self.enc1(x)  # (B, base_ch, F, W, H)
        logger.debug(f"After initial encoder: {h.shape}")
        enc_features = [h]
        for enc in self.next_encoders:
            h = self.avg_pool(h)  # Downsample
            h = enc(h)
            logger.debug(f"After encoder layer: {h.shape}")
            enc_features.append(h)
        for i, dec in enumerate(self.init_dec):
            h = self.interpolate(h)  # Upsample
            h = torch.cat([h, enc_features[-(i + 2)]], dim=1)  # Skip connection
            h = dec(h)
            logger.debug(f"After decoder layer: {h.shape}")
        out = self.out_dec(h)  # (B, in_ch, F, W, H)
        logger.debug(f"After decoder layer: {out.shape}")

        assert out.shape == (B, C, F, W, H), f"Output shape mismatch: expected {(B, C, F, W, H)}, got {out.shape}"

        # Remove the init frames from the output
        # out = out[:, :, x_init.shape[2]:, :, :]  # (B, C, F_out, W, H)

        logger.debug(f"Forward pass with out shape: {out.shape}")
        return out

    def generation(self, x, x_init, n_euler_steps, t_start=0.0, t_end=1.0):
        logger.debug(f"Generation with x shape: {x.shape}, n_euler_steps: {n_euler_steps}, t_start: {t_start}, t_end: {t_end}")
        time_steps = torch.linspace(t_start, t_end, n_euler_steps + 1).view(-1, 1, 1, 1, 1).to(x.device)

        for i in range(n_euler_steps):
            x = x + (time_steps[i + 1] - time_steps[i]) * self(t=time_steps[i], x_t=x, x_init=x_init)

        logger.debug(f"Generation with x shape: {x.shape}")
        return x
