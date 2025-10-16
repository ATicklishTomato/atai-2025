from os import listdir, path, pardir

import numpy as np

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from tqdm import trange


# CONSTANTS 
DOMAIN_SIZE = 1000
DATA_FOLDER = path.join(pardir, 'data', 'boids')
RAW_DATA_FOLDER = path.join(DATA_FOLDER, 'raw', '')
PROCESSED_DATA_FOLDER = path.join(DATA_FOLDER, 'processed', '')


class BoidsDataset(InMemoryDataset):
    def __init__(self, raw_data_path, processed_data_path, root=None, transform=None, pre_transform=None, post_transform=None, solution_idx_range=(0, 25), timesteps=1000, processed_file_name="AR3_Boids.pt"):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.solution_idx_range = solution_idx_range
        self.timesteps = timesteps
        self.processed_file_name = processed_file_name
        self.pre_transform = pre_transform
        self.transform = transform
        self.post_transform = post_transform
        super(BoidsDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return [self.processed_file_name]

    @property
    def raw_file_names(self):
        return [pfn for pfn in listdir(self.raw_data_path) if (self.solution_idx_range[0] <= int(pfn.split("_")[-1][:-4]) < self.solution_idx_range[1])]
    
    def download(self):
        pass
    
    def __len__(self):
        return (self.timesteps - 1) * (self.solution_idx_range[1] - self.solution_idx_range[0])

    def process(self):
        positions_list = []
        data_list = []
        for idx, raw_path in enumerate(self.raw_file_names):
            trajectory = np.load(self.raw_data_path + raw_path)

            if self.transform is not None:
                trajectory = self.transform(trajectory)
                
            # Add the initial positions to the positions list
            positions_list.append(trajectory[0, :, :2])

            for t in trange(trajectory.shape[0] - 1):
                x = torch.tensor(trajectory[t], dtype=torch.float)
                y = torch.tensor(trajectory[t+1], dtype=torch.float)
                
                # Create fully connected graph
                n = trajectory.shape[1]
                edge_index = torch.tensor([[i, j] for i in range(n) for j in range(n) if i != j], dtype=torch.long).t().contiguous()
                
                data = Data(x=x, y=y, edge_index=edge_index)
                if self.post_transform is not None:
                    data = self.post_transform(data)
                
                data_list.append(data)
                
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_data_path+self.processed_file_name)
        torch.save(torch.tensor(positions_list), self.processed_data_path+"positions_"+self.processed_file_name)

    def __getitem__(self, idx):
        return self.get(idx)
    
    def __repr__(self):
        return f'{self.__class__.__name__}({len(self)})'


def get_boids_datasets():
    train_dataset = BoidsDataset(
        raw_data_path=RAW_DATA_FOLDER, 
        processed_data_path=PROCESSED_DATA_FOLDER, 
        root=DATA_FOLDER, 
        solution_idx_range=(0, 15), 
        timesteps=1000, 
        processed_file_name="AR3_Boids_Equivariant.pt",
        transform=lambda traj: traj / DOMAIN_SIZE
    )

    validation_dataset = BoidsDataset(
        raw_data_path=RAW_DATA_FOLDER, 
        processed_data_path=PROCESSED_DATA_FOLDER, 
        root=DATA_FOLDER, 
        solution_idx_range=(16, 25), 
        timesteps=1000, 
        processed_file_name="AR3_VAL_Boids_Equivariant.pt",
        transform=lambda traj: traj / DOMAIN_SIZE
    )

    return train_dataset, validation_dataset

def get_initial_states_boids_validation_dataset():
    def keep_01(data):
        return data[0:2, :, :] / DOMAIN_SIZE

    initial_states_validation_dataset = BoidsDataset(
        raw_data_path=RAW_DATA_FOLDER, 
        processed_data_path=PROCESSED_DATA_FOLDER, 
        root=DATA_FOLDER, 
        solution_idx_range=(16, 25), 
        timesteps=2, 
        processed_file_name="AR1_VAL_init.pt",
        transform=keep_01
    )

    return initial_states_validation_dataset
