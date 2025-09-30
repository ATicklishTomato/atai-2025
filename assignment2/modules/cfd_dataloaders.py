import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os

class CFDDataset(Dataset):
    # TODO: Implement the CFD dataset class
    def __init__(self, filenames, flip_augmentation=False, timesample=1, bundle=1):
        self.sequences = []
        self.index_map = []
        self.flip_augmentation = flip_augmentation
        self.bundle = bundle

        # coordinates
        self.coordsy = np.linspace(-5, 5, 64, endpoint=True)
        self.coordsx = np.linspace(-10, 10, 128, endpoint=True)
        self.coords = np.array(np.meshgrid(self.coordsx, self.coordsy)).T.reshape(128, 64, 2)
        self.coords = torch.tensor(self.coords, dtype=torch.float32).permute(2, 1, 0).cuda()

        # obstacle mask
        center = torch.tensor([-5.0, 0.0], device=self.coords.device).view(2, 1, 1)
        radius = 0.5
        squared_distance = ((self.coords - center) ** 2).sum(dim=0)
        self.mask = (squared_distance < radius**2).unsqueeze(0).cuda()  # [1, 64, 128]

        # load sequences
        for seq_idx, filename in enumerate(filenames):
            data = np.load(filename)  # shape [T, C, H, W]
            data = data[::timesample]  # subsample in time
            self.sequences.append(data)
            T = data.shape[0]
            # only keep indices where bundle+1 frames are available
            self.index_map.extend([(seq_idx, t) for t in range(T - bundle)])

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        seq_idx, t = self.index_map[idx]
        seq = self.sequences[seq_idx]

        # time-bundled input [bundle, C, H, W]
        input_seq = seq[t:t + self.bundle]
        target = seq[t + self.bundle]  # predict next frame

        # optional flip augmentation
        if self.flip_augmentation and np.random.rand() > 0.5:
            input_seq = np.stack([self.flip(x) for x in input_seq], axis=0)
            target = self.flip(target)

        return (
            self.mask,
            self.coords,
            torch.tensor(input_seq, dtype=torch.float32),  # [bundle, C, H, W]
            torch.tensor(target, dtype=torch.float32)      # [C, H, W]
        )

    def get_trajectory(self, seq_idx):
        seq = self.sequences[seq_idx]
        return (
            self.mask.unsqueeze(0),
            self.coords.unsqueeze(0),
            torch.tensor(seq, dtype=torch.float32)  # [T, C, H, W]
        )

    def flip(self, x):
        x = np.flip(x, axis=2).copy()
        x[1] *= -1
        return x

def preprocess_files():
    if not os.path.exists("./data/cfd/processed/"):
        os.makedirs("./data/cfd/processed/", exist_ok=True)
    if os.listdir("./data/cfd/processed/") != []:
        return

    os.makedirs("./data/cfd/processed/", exist_ok=True)
    for Re in [100, 150, 200, 250, 300, 350, 400]:
        u = np.load(f"./data/cfd/raw/u_grid_Re{Re}.npy")
        v = np.load(f"./data/cfd/raw/v_grid_Re{Re}.npy")
        p = np.load(f"./data/cfd/raw/p_grid_Re{Re}.npy")
        concat = np.stack([u, v, p], axis=1)
        filename_save = f"./data/cfd/processed/uvp_grid_Re{Re}.npy"
        np.save(filename_save, concat)



def get_cfd_dataloaders(dt=20, bundle=5, batch_size=1, train_files=None, val_files=None):
    if train_files is None or len(train_files) == 0:
        train_files = [
            './data/cfd/processed/uvp_grid_Re100.npy',
            './data/cfd/processed/uvp_grid_Re200.npy',
            './data/cfd/processed/uvp_grid_Re300.npy',
            './data/cfd/processed/uvp_grid_Re400.npy'
        ]
    if val_files is None or len(val_files) == 0:
        val_files = [
            './data/cfd/processed/uvp_grid_Re150.npy',
            './data/cfd/processed/uvp_grid_Re250.npy',
            './data/cfd/processed/uvp_grid_Re350.npy'
        ]

    preprocess_files()

    train_dataset = CFDDataset(train_files, flip_augmentation=False, timesample=dt, bundle=bundle)
    val_dataset = CFDDataset(val_files, flip_augmentation=False, timesample=dt, bundle=bundle)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader