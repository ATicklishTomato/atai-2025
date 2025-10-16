import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple
from torch import Tensor

class CFDBaselineDataset(Dataset):
    def __init__(self, filenames, flip_augmentation=False, timesample=1):
        self.sequences = []
        self.index_map = []
        self.flip_augmentation = flip_augmentation

        # coordinates
        self.coordsy = np.linspace(-5, 5, 64, endpoint=True)
        self.coordsx = np.linspace(-10, 10, 128, endpoint=True)
        self.coords = np.array(np.meshgrid(self.coordsx, self.coordsy)).T.reshape(128, 64, 2)
        self.coords = torch.tensor(self.coords, dtype=torch.float32).permute(2, 1, 0).cuda()

        # use coordinates to make obstacle mask
        center = torch.tensor([-5.0, 0.0], device=self.coords.device).view(2, 1, 1)
        radius = 0.5
        squared_distance = ((self.coords - center) ** 2).sum(dim=0) 
        self.mask = squared_distance < radius**2  # shape [64, 128]
        self.mask = self.mask.unsqueeze(0).cuda()

        # sample/read the data
        for seq_idx, filename in enumerate(filenames):
            data = np.load(filename)  
            data = data[::timesample] 
            self.sequences.append(data)
            T = data.shape[0]
            self.index_map.extend([(seq_idx, t) for t in range(T - 1)])

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        seq_idx, t = self.index_map[idx]
        seq = self.sequences[seq_idx]
        input = seq[t]    
        target = seq[t + 1] 

        # if flip augmentation is true then flip the data horizontally 50% of the time
        if self.flip_augmentation and np.random.rand() > 0.5:
            input = self.flip(input)
            target = self.flip(target)
        return (
                self.mask, 
                self.coords, 
                torch.tensor(input, dtype=torch.float32), 
                torch.tensor(target, dtype=torch.float32)
                )
        
    def get_trajectory(self, seq_idx):
        # returns full trajectory
        seq = self.sequences[seq_idx]
        return (
            self.mask.unsqueeze(0), 
            self.coords.unsqueeze(0), 
            torch.tensor(seq, dtype=torch.float32)
        )

    def flip(self, x):
        x = np.flip(x, axis=2).copy()
        x[1] *= -1
        return x

    def get_step_data_points(self, steps: int) -> List[Tuple[Tensor, Tensor]]:
        """
        """
        assert steps > 0, "Number of steps must be positive."
        
        # Maximum allowed rollout steps such that a target exists T steps ahead
        max_steps = self.get_maximum_step_size()
        assert steps <= max_steps, "Number of steps must be less than or equal to the maximum step size."

        data_points: List[Tuple[Tensor, Tensor]] = []
        
        # Prepare mask channel on CPU once (same spatial size as data)
        mask_channel = self.mask.to('cpu').float()  # [1, H, W]

        for seq in self.sequences:
            # seq shape: [F, C, H, W] with C=3 (u, v, p)
            total_frames = seq.shape[0]

            # Valid start indices so that target at t+steps exists
            for t in range(0, total_frames - steps):
                inp_np = np.array(seq[t])            # [C, H, W]
                targ_np = np.array(seq[t + steps])   # [C, H, W]

                # Convert to tensors on CPU
                inp = torch.tensor(inp_np, dtype=torch.float32)   # [3, H, W]
                targ = torch.tensor(targ_np, dtype=torch.float32) # [3, H, W]

                # Concatenate mask as an additional channel for the input to match in_channels=4
                inp_with_mask = torch.cat([inp, mask_channel], dim=0)  # [4, H, W]
                targ_with_mask = torch.cat([targ, mask_channel], dim=0)  # [4, H, W]

                # Add batch dimension
                input_state = inp_with_mask.unsqueeze(0)   # [1, 4, H, W]
                target_state = targ_with_mask.unsqueeze(0) # [1, 4, H, W]

                data_points.append((input_state, target_state))

        return data_points

    def get_maximum_step_size(self):
        """
        """
        # All trajectories have identical length. The maximum number of rollout
        # steps such that a target exists for i=0 is (T - 1).
        last_frame_index = self.sequences[0].shape[0] - 1
        return last_frame_index



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropprob=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(p=dropprob)
        )

    def forward(self, x):
        return self.net(x)

class CFDBaselineModel(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64,mult=[1, 2, 4, 8]):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, base_channels * mult[0])
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_channels * mult[0], base_channels * mult[1])
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_channels * mult[1], base_channels * mult[2])
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * mult[2], base_channels * mult[3])

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_channels * mult[3], base_channels * mult[2], kernel_size=2, stride=2)
        self.dec3 = DoubleConv(2 *base_channels * mult[2], base_channels * mult[2])
        self.up2 = nn.ConvTranspose2d(base_channels * mult[2], base_channels * mult[1], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(2 * base_channels * mult[1], base_channels * mult[1])
        self.up1 = nn.ConvTranspose2d(base_channels * mult[1], base_channels * mult[0], kernel_size=2, stride=2)
        self.dec1 = DoubleConv(2 * base_channels * mult[0], base_channels * mult[0])
        
        self.out_conv = nn.Conv2d(base_channels * mult[0], out_channels, kernel_size=1)

    def forward(self, x):

        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))

        x4 = self.bottleneck(self.pool3(x3))

        x = self.up3(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)
        x = self.up2(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)
        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)

        return self.out_conv(x)
    
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
