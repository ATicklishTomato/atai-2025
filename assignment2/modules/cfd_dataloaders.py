import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import logging

from typing import List, Tuple
from torch import Tensor

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

    def get_step_data_points(self, steps: int) -> List[Tuple[Tensor, Tensor]]:
        """
        Get the all pairs of data points within the same sequence that have an initial frame and a target frame separated by `steps` frames.
        Thus we want the List[(start_state, target_state)] where target_state is `steps` frames after the start_state.

        We use these data points to evaluate the peformance of the model on predicting `steps` steps.
        When this method is called with the maximum step size it should return the initial state of the trajectory (so one data point per trajectory).
        
        Ensure that we return data points (input, target) of shape ([1, C+1, F_hist, W, H], [1, C+1, 1, W, H]), such that
        the index of the target frame is at position (i + (F_hist-1) + steps) if the index of the first input frame is at index i.
        """
        assert steps > 0, "Number of steps must be positive."

        max_steps = self.get_maximum_step_size()
        assert steps <= max_steps, "Number of steps must be less than or equal to the maximum step size."

        data_points = []
        for seq in self.sequences:
            last_frame_index = len(seq) - 1
            # The last frame is at index last_frame_index, hence the last start index
            # is when (i + (F_hist-1) + steps) = last_frame_index, hence the first start index
            # is for i = last_frame_index - (F_hist-1) - steps
            start_indices = range(0, last_frame_index - (self.history_frames - 1) - steps)
            for i in start_indices:
                input_window = np.array(seq[i:i + self.history_frames])
                target = np.array(seq[i + (self.history_frames - 1) + steps])

                # Convert to [C, F, W, H]
                input_seq = np.transpose(input_window, (1, 0, 2, 3))
                target = np.transpose(target, (1, 0, 2, 3))

                # Build mask channels on CPU and concatenate along channel dim
                input_mask = self.mask.to('cpu').repeat(1, self.history_frames, 1, 1)
                target_mask = self.mask.to('cpu').repeat(1, 1, 1, 1)

                input_tensor = torch.tensor(input_seq, dtype=torch.float32)
                target_tensor = torch.tensor(target, dtype=torch.float32)

                input_state = torch.cat([input_mask, input_tensor], dim=0).unsqueeze(0)   # [1, C+1, F_hist, W, H]
                target_state = torch.cat([target_mask, target_tensor], dim=0).unsqueeze(0) # [1, C+1, 1, W, H]

                data_points.append((input_state, target_state))

        return data_points

    def get_maximum_step_size(self):
        """
        Get the maximum `step` size, such that there is one data point per sequence.
        This is the maximum number of frames that can be predicted such that a ground truth target state is available.

        The maximum step size is then the number for steps such that (i + (F_hist-1) + steps) = last_frame_index for i = 0.
        Hence the maximum step size is last_frame_index - (F_hist-1).
        """
        assert len(self.sequences) > 0, "No sequences loaded."

        # The lengths of each of the sequences are the same
        last_frame_index = len(self.sequences[0]) - 1

        max_step = last_frame_index - (self.history_frames - 1)
        
        # Ensure there is at least one valid step
        assert max_step > 0, "Sequences are too short for any positive step size."
        return max_step

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