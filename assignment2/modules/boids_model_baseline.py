import torch
import torch.nn as nn
from torch_geometric.data import Data
import numpy as np
from typing import List, Tuple

# Re-export the BoidsDataset from boids_dataloaders for convenience
# from modules.boids_dataloaders import BoidsDataset

DOMAIN_SIZE = 1000.0

class BoidsBaselineDataset:
    def __init__(self, filenames, timesample=1):
        self.sequences = []
        self.index_map = []
        
        # Load trajectories
        for seq_idx, filename in enumerate(filenames):
            data = np.load(filename)  # Shape: [timesteps, boids, 4]
            data = data[::timesample]
            self.sequences.append(data)
            T = data.shape[0]
            self.index_map.extend([(seq_idx, t) for t in range(T - 1)])
    
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        seq_idx, t = self.index_map[idx]
        seq = self.sequences[seq_idx]
        
        x = torch.tensor(seq[t], dtype=torch.float32)    # [25, 4]
        y = torch.tensor(seq[t + 1], dtype=torch.float32)  # [25, 4]
        
        # Create fully connected edge index (excluding self-loops)
        num_nodes = x.shape[0]
        edge_index = torch.tensor(
            [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j],
            dtype=torch.long
        ).t().contiguous()
        
        return Data(x=x, y=y, edge_index=edge_index)
    
    def get_step_data_points(self, steps: int) -> List[Tuple[Data, Data]]:
        """
        Returns list of (input_data, target_data) tuples for the given step size.
        Both input and target are Data objects.
        """
        assert steps > 0, "Number of steps must be positive."
        max_steps = self.get_maximum_step_size()
        assert steps <= max_steps, f"Steps {steps} exceeds maximum {max_steps}"
        
        data_points = []
        
        for seq in self.sequences:
            total_frames = seq.shape[0]
            num_nodes = seq.shape[1]
            
            # Create edge index once (same for all frames)
            edge_index = torch.tensor(
                [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j],
                dtype=torch.long
            ).t().contiguous()
            
            for t in range(total_frames - steps):
                inp = torch.tensor(seq[t], dtype=torch.float32)
                targ = torch.tensor(seq[t + steps], dtype=torch.float32)
                
                input_data = Data(x=inp, edge_index=edge_index)
                target_data = Data(x=targ, edge_index=edge_index)
                
                data_points.append((input_data, target_data))
        
        return data_points
    
    def get_maximum_step_size(self):
        """Returns maximum number of rollout steps."""
        # All trajectories have same length
        return self.sequences[0].shape[0] - 1

def square_torus_distance(p1, p2, width=1.0, height=1.0):
    """
    Compute the squared torus distance between two points p1 and p2 in a box of given width and height.
    
    Args:
        p1: Tensor of shape (num_points, 2) representing the first point(s)
        p2: Tensor of shape (num_points, 2) representing the second point(s)
        width: Width of the box
        height: Height of the box
    Returns:
        dist: Tensor of shape (num_points,) representing the squared torus distance between p1 and p2
    """
    dx = p1[:, 0] - p2[:, 0]
    dy = p1[:, 1] - p2[:, 1]
    dx = dx - width * torch.round(dx / width)
    dy = dy - height * torch.round(dy / height)
    return dx ** 2 + dy ** 2

def pbc_direction(p1, p2):
    """
    Compute the direction from p1 to p2 considering periodic boundary conditions in a unit square.
    Args:
        p1: Tensor of shape (num_points, 2) representing the first point(s)
        p2: Tensor of shape (num_points, 2) representing the second point(s)
    Returns:
        direction: Tensor of shape (num_points, 2) representing the direction from p1 to p2 considering PBCs    
    """
    return p2 - p1 - torch.round(p2 - p1)


class EGNNLayer(nn.Module):
    def __init__(self, h_dim=16, m_dim=16, hidden_dim=16):
        super(EGNNLayer, self).__init__()

        # Constants
        self.edge_dim = 3
        self.width = 1.0
        self.height = 1.0

        # Networks
        self.phi_e = nn.Sequential(
            nn.Linear(h_dim * 2 + self.edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, m_dim),
            nn.SiLU()
        )
        self.phi_v = nn.Sequential(
            nn.Linear(m_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        phi_x_last_layer = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.xavier_uniform_(phi_x_last_layer.weight, gain=0.001)
        self.phi_x = nn.Sequential(
            nn.Linear(m_dim, hidden_dim),
            nn.SiLU(),
            phi_x_last_layer
        )

        self.node_embedding_nn = nn.Sequential(
            nn.Linear(h_dim + m_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, h_dim)
        )

    def phi_h(self, h, m):
        agg = torch.cat([h, m], dim=1)
        out = self.node_embedding_nn(agg)
        return out + h

    def calculate_edge_attributes(self, x, edge_index):
        edge_attr = torch.zeros((edge_index.shape[1], self.edge_dim), dtype=torch.float, device=x.device)

        # Square torus distance (to get distance between boids with PBCs)
        edge_attr[:, 0] = square_torus_distance(x[edge_index[0], :2], x[edge_index[1], :2], width=self.width, height=self.height)

        # Cosine similarity of velocities (to get alignment of velocities)
        edge_attr[:, 1] = nn.functional.cosine_similarity(x[edge_index[0], 2:], x[edge_index[1], 2:], dim=1)

        # Calculate l2 norm between velocities (to get speed difference)
        edge_attr[:, 2] = torch.norm(x[edge_index[1], 2:] - x[edge_index[0], 2:], dim=1).pow(2)

        return edge_attr

    def forward(self, x, edge_index, h, v_init):
        edge_index = edge_index[:, square_torus_distance(x[edge_index[0], :2], x[edge_index[1], :2], width=self.width, height=self.height) < 0.04]
        if edge_index.numel() == 0:
            x[:, 2:] = self.phi_v(h) * v_init
            x[:, :2] += x[:, 2:]
            x[:, 0] %= self.width
            x[:, 1] %= self.height
            return h, x

        # Calculate edge features
        mij = self.phi_e(torch.cat([h[edge_index[0]], h[edge_index[1]], self.calculate_edge_attributes(x, edge_index).detach()], dim=1))

        # Update velocity
        C = 1.0
        force_sum = torch.zeros_like(x[:, 2:])

        # Add repulsion/attraction
        x[:, 2:] = self.phi_v(h) * v_init + C * force_sum.scatter_add(
            0, edge_index[0].unsqueeze(-1).expand(-1, 2), pbc_direction(x[edge_index[0], :2], x[edge_index[1], :2]) * self.phi_x(mij)
        )

        # Update position
        x[:, :2] += x[:, 2:]
        x[:, 0] %= self.width
        x[:, 1] %= self.height
        
        # Update h
        mi = torch.zeros_like(h)
        mi = mi.scatter_add(0, edge_index[0].unsqueeze(-1).expand(-1, mij.size(1)), mij)
        h = self.phi_h(h, mi)

        return h, x


class BoidsBaselineModel(nn.Module):
    def __init__(self, h_dim=16, m_dim=16, hidden_dim=16, layers=1):
        super(BoidsBaselineModel, self).__init__()
        self.h_dim = h_dim
        self.embedding = nn.Linear(1, h_dim)
        self.layers = nn.ModuleList([EGNNLayer(h_dim, m_dim, hidden_dim) for _ in range(layers)])

    def forward(self, data):
        """Core forward pass expecting a Data object."""
        x, edge_index = data.x.clone().detach(), data.edge_index

        x /= DOMAIN_SIZE
        v_init = x[:, 2:].clone().detach()
        h = self.embedding(torch.norm(x[:, 2:], dim=1, keepdim=True))
        for mp_layer in self.layers:
            h, x = mp_layer(x, edge_index, h, v_init)
        x *= DOMAIN_SIZE

        return x