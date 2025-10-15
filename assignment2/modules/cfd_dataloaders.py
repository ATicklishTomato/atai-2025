import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import logging

logger = logging.getLogger(__name__)

class CFDDataset(Dataset):
    def __init__(self, filenames, flip_augmentation=False, timesample=1, predict_frames=1, history_frames=5):
        self.sequences = []
        self.index_map = []
        self.flip_augmentation = flip_augmentation
        self.predict_frames = predict_frames
        self.history_frames = history_frames

        # coordinates
        self.coordsy = np.linspace(-5, 5, 64, endpoint=True)
        self.coordsx = np.linspace(-10, 10, 128, endpoint=True)
        self.coords = np.array(np.meshgrid(self.coordsx, self.coordsy)).T.reshape(128, 64, 2)
        self.coords = torch.tensor(self.coords, dtype=torch.float32).permute(2, 1, 0).cuda()

        # obstacle mask
        center = torch.tensor([-5.0, 0.0], device=self.coords.device).view(2, 1, 1)
        radius = 0.5
        squared_distance = ((self.coords - center) ** 2).sum(dim=0)
        self.mask = (squared_distance < radius**2).unsqueeze(0).cuda()  # [C, W, H]

        # load sequences
        for seq_idx, filename in enumerate(filenames):
            data = np.load(filename)  # shape [F, C, H, W]
            data = data[::timesample]  # subsample in time
            self.sequences.append(data)
            total_frames = data.shape[0]
            # only keep indices where we have enough history and prediction frames
            self.index_map.extend([(seq_idx, t) for t in range(history_frames, total_frames - predict_frames)])

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        """ Prepares and returns a data sample.
        @param idx: Index of the sample to retrieve.
        @return: A tuple containing:
        - history_mask: Mask for the history frames [1, F, W, H]
        - history_sequence: History frames [C, F, W, H]
        - target_mask: Mask for the target frames [1, F, W, H]
        - target_sequence: Target frames [C, F, W, H]

        """
        seq_idx, t = self.index_map[idx]
        seq = self.sequences[seq_idx]

        # time-bundled data [F, C, W, H]
        original_target_sequence = np.array(seq[t:t + self.predict_frames])  # [F, C, W, H]
        original_history_sequence = np.array(seq[t-self.history_frames:t])
        # Put channels first, such that [C, F, W, H]
        target_sequence = np.transpose(original_target_sequence, (1, 0, 2, 3))  # [C, F, W, H]
        history_sequence = np.transpose(original_history_sequence, (1, 0, 2, 3))  # [C, F, W, H]

        for channel in range(target_sequence.shape[0]):
            for frame in range(target_sequence.shape[1]):
                target = target_sequence[channel, frame]
                true = original_target_sequence[frame, channel]
                assert np.array_equal(target, true), f"Data mismatch at seq {seq_idx}, time {t}, channel {channel}, frame {frame}"


        # Resize masks to respective [F, 1, W, H] sizes for history and target sequences
        history_mask = self.mask.repeat(1, self.history_frames, 1, 1)
        target_mask = self.mask.repeat(1, self.predict_frames, 1, 1)


        if self.flip_augmentation and np.random.rand() > 0.5:
            history_sequence = self.flip(history_sequence)
            target_sequence = self.flip(target_sequence)
            history_mask = self.flip(history_mask)
            target_mask = self.flip(target_mask)

        # Make tensors
        history_sequence = torch.tensor(history_sequence, dtype=torch.float32)
        target_sequence = torch.tensor(target_sequence, dtype=torch.float32)

        return history_mask, history_sequence, target_mask, target_sequence


    def get_trajectory(self, seq_idx):
        seq = self.sequences[seq_idx]
        seq = np.transpose(seq, (1, 0, 2, 3))  # [C, F, W, H]
        seq = torch.tensor(seq, dtype=torch.float32)
        mask = self.mask.repeat(1, self.history_frames, 1, 1) # [1, F, W, H]
        return mask, seq


    def flip(self, x):
        """ Flips the input tensor horizontally and negates the x-component of the velocity."""
        x = np.flip(x, axis=3)  # flip width dimension
        x[1] *= -1 # negate x-velocity
        return x

def preprocess_files():
    logger.info('Preprocessing files')
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



def get_cfd_dataloaders(dt=20, predict_frames=20, history_frames=5, batch_size=1, train_files=None, val_files=None):
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

    logger.info('Building datasets')
    train_dataset = CFDDataset(train_files, flip_augmentation=False, timesample=dt, predict_frames=predict_frames,
                                history_frames=history_frames)
    val_dataset = CFDDataset(val_files, flip_augmentation=False, timesample=dt, predict_frames=predict_frames,
                                history_frames=history_frames)

    logger.info('Building dataloaders')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.info('Finished loading data')
    return train_loader, val_loader