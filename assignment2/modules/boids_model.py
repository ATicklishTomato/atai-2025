import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.utils import scatter, add_self_loops, degree

from .boids_helper_functions import square_torus_distance, pbc_direction


class GeometricFlowMatchingModel(torch.nn.Module):
    def __init__(self):
        super(GeometricFlowMatchingModel, self).__init__()

    def forward(self, t: Tensor, data_t: Data, data_c: Data) -> Tensor:
        raise NotImplementedError("This method should be implemented in a subclass")

    def generate(self, data_c: Data, data_0: Data, n_euler_steps: int, t_start=0.0, t_end=1.0):
        time_steps = torch.linspace(t_start, t_end, n_euler_steps + 1).to(data_0.x.device)
        data_t = data_0.clone()

        for i in range(n_euler_steps):
            data_t.x += (time_steps[i+1] - time_steps[i]) * self(t=time_steps[i].unsqueeze(-1), data_t=data_t, data_c=data_c)
            data_t.x[:, :2] %= 1.0

        return data_t.x


class EGNNLayer(torch.nn.Module):
    def __init__(self, h_dim=16, m_dim=16, hidden_dim=16):
        super(EGNNLayer, self).__init__()

        # Constants
        self.edge_dim = 3
        self.width = 1.0
        self.height = 1.0
        STD = 1e-5

        # edge messages
        self.phi_e = torch.nn.Sequential(
            torch.nn.Linear(h_dim * 2 + self.edge_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, m_dim),
            torch.nn.SiLU()
        )
        torch.nn.init.normal_(self.phi_e[2].weight, mean=0, std=STD)

        # speed
        self.phi_v = torch.nn.Sequential(
            torch.nn.Linear(m_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
        torch.nn.init.normal_(self.phi_v[2].weight, mean=0, std=STD)

        # force
        phi_x_last_layer = torch.nn.Linear(hidden_dim, 1, bias=False)
        torch.nn.init.xavier_uniform_(phi_x_last_layer.weight, gain=0.001)
        self.phi_x = torch.nn.Sequential(
            torch.nn.Linear(m_dim, hidden_dim),
            torch.nn.SiLU(),
            phi_x_last_layer
        )

        # new hidden state
        self.node_embedding_nn = torch.nn.Sequential(
            torch.nn.Linear(h_dim + m_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, h_dim)
        )
        torch.nn.init.normal_(self.node_embedding_nn[2].weight, mean=0, std=STD)
    
    def phi_h(self, h, m):
        agg = torch.cat([h, m], dim=1)
        out = self.node_embedding_nn(agg)
        return out + h

    def forward(self, x, edge_index, edge_attr, h, v_init):
        # Calculate edge features
        mij = self.phi_e(torch.cat([h[edge_index[0]], h[edge_index[1]], edge_attr], dim=1))

        # Add repulsion/attraction
        x[:, 2:] = self.phi_v(h) * v_init + scatter(
            pbc_direction(x[edge_index[0], :2], x[edge_index[1], :2]) * self.phi_x(mij), 
            edge_index[0], 
            0, 
            dim_size=x.shape[0], 
            reduce='sum'
        )

        # Update position
        x[:, :2] += x[:, 2:]
        x[:, 0] %= self.width
        x[:, 1] %= self.height
        
        # Update h
        h = self.phi_h(h, scatter(mij, edge_index[0], 0, dim_size=h.size(0), reduce='sum'))

        return h, x


class BoidsModel(GeometricFlowMatchingModel):
    def __init__(self, h_dim=16, m_dim=16, hidden_dim=16, layers=1):
        super(BoidsModel, self).__init__()
        self.h_dim = h_dim
        self.embedding = torch.nn.Linear(4, h_dim)
        self.layers = torch.nn.ModuleList([EGNNLayer(h_dim, m_dim, hidden_dim) for _ in range(layers)])

    def calculate_edge_attributes(self, x, edge_index):
        edge_attr = torch.zeros((edge_index.shape[1], 3), dtype=torch.float, device=x.device)

        # Square torus distance (to get distance between boids with PBCs)
        edge_attr[:, 0] = square_torus_distance(x[edge_index[0], :2], x[edge_index[1], :2])

        # Cosine similarity of velocities (to get alignment of velocities)
        edge_attr[:, 1] = torch.nn.functional.cosine_similarity(x[edge_index[0], 2:], x[edge_index[1], 2:], dim=1)

        # Calculate l2 norm between velocities (to get speed difference)
        edge_attr[:, 2] = (torch.norm(x[edge_index[1], 2:], dim=1) - torch.norm(x[edge_index[0], 2:], dim=1)).pow(2)

        return edge_attr

    def forward(self, t: Tensor, data_t: Data, data_c: Data):
        x, edge_index = data_t.x.clone().detach(), data_t.edge_index
        x_init = x.clone()

        # Restrict edges
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])
        edge_weights = square_torus_distance(x[edge_index[0], :2], x[edge_index[1], :2])
        edge_mask = edge_weights <= 0.04 ** 2
        edge_index = edge_index[:, edge_mask]
        edge_attr = self.calculate_edge_attributes(x, edge_index)

        # Create hidden layer
        h = torch.cat((t.expand(x.shape[:-1]).unsqueeze(-1), x[:, 2:], degree(edge_index[0], num_nodes=x.shape[0]).view(-1, 1)), dim=1)
        h = self.embedding(h)

        # Message passing
        for mp_layer in self.layers:
            h, x = mp_layer(x, edge_index, edge_attr, h, x_init[:, 2:])

        # Create output
        out = torch.zeros_like(x)
        out[:, :2] = pbc_direction(x_init[:, :2], x[:, :2])
        out[:, 2:] = x[:, 2:] - x_init[:, 2:]

        return out
