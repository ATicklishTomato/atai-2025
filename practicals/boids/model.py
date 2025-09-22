import torch 
import torch_geometric
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from collections import defaultdict
from contextlib import contextmanager


class HierarchicalTimer:
    """
    A timer that supports nested timing contexts with hierarchical tracking.
    
    Usage:
        timer = HierarchicalTimer()
        
        with timer.time("training"):
            with timer.time("data_loading"):
                # data loading code
                pass
            with timer.time("forward_pass"):
                # forward pass code
                pass
                
        timer.print_timings()
        timer.reset()
    """
    
    def __init__(self):
        self.timings = defaultdict(float)  # Flat storage: "key1.key2.key3" -> total_time
        self.counts = defaultdict(int)     # How many times each key was called
        self.stack = []                    # Current timing stack
        self.start_times = {}              # Start time for each active timing
        
    @contextmanager
    def time(self, key):
        """Context manager for timing a section of code"""
        # Build hierarchical key
        full_key = ".".join(self.stack + [key])
        
        # Start timing
        self.stack.append(key)
        start_time = time.time()
        self.start_times[full_key] = start_time
        
        try:
            yield
        finally:
            # End timing
            end_time = time.time()
            elapsed = end_time - start_time
            
            self.timings[full_key] += elapsed
            self.counts[full_key] += 1
            
            # Clean up
            if full_key in self.start_times:
                del self.start_times[full_key]
            self.stack.pop()
    
    def print_timings(self, min_time=0.001, show_tree=True):
        """Print timings in a hierarchical format"""
        if not self.timings:
            print("No timings recorded.")
            return
        
        if show_tree:
            self._print_tree_format(min_time)
        else:
            self._print_level_format(min_time)
    
    def _print_tree_format(self, min_time=0.001):
        """Print timings in a tree format"""
        print("\n" + "="*80)
        print("HIERARCHICAL TIMING REPORT (Tree View)")
        print("="*80)
        
        # Build tree structure
        tree = {}
        for key, total_time in self.timings.items():
            if total_time >= min_time:
                parts = key.split('.')
                current = tree
                for part in parts:
                    if part not in current:
                        current[part] = {'_children': {}, '_time': 0, '_count': 0}
                    current = current[part]['_children']
                # Store timing info at the leaf
                leaf = tree
                for part in parts[:-1]:
                    leaf = leaf[part]['_children']
                leaf[parts[-1]]['_time'] = total_time
                leaf[parts[-1]]['_count'] = self.counts[key]
        
        # Print tree recursively
        def print_node(node_dict, prefix="", is_last=True):
            items = list(node_dict.items())
            for i, (key, value) in enumerate(items):
                is_last_item = (i == len(items) - 1)
                
                # Current prefix for this line
                current_prefix = "└── " if is_last_item else "├── "
                
                # Format timing info
                time_val = value['_time']
                count_val = value['_count']
                if time_val > 0:
                    avg_time = time_val / count_val if count_val > 0 else 0
                    timing_info = f"{time_val:8.4f}s total | {avg_time:8.4f}s avg | {count_val:4d} calls"
                    print(f"{prefix}{current_prefix}{key:<25} | {timing_info}")
                else:
                    print(f"{prefix}{current_prefix}{key}")
                
                # Prepare prefix for children
                if value['_children']:
                    child_prefix = prefix + ("    " if is_last_item else "│   ")
                    print_node(value['_children'], child_prefix, is_last_item)
        
        print_node(tree)
        
        # Calculate total time
        total_time = sum(time for key, time in self.timings.items() if '.' not in key)
        print("-" * 80)
        print(f"TOTAL TOP-LEVEL TIME: {total_time:.4f}s")
        print("="*80)
    
    def _print_level_format(self, min_time=0.001):
        """Print timings grouped by hierarchy level (original format)"""
        # Group by hierarchy level
        by_level = defaultdict(list)
        for key, total_time in self.timings.items():
            if total_time >= min_time:  # Filter out very small timings
                level = key.count('.')
                by_level[level].append((key, total_time))
        
        # Sort each level by time (descending)
        for level in by_level:
            by_level[level].sort(key=lambda x: x[1], reverse=True)
        
        print("\n" + "="*60)
        print("HIERARCHICAL TIMING REPORT (Level View)")
        print("="*60)
        
        # Print level by level
        for level in sorted(by_level.keys()):
            if level == 0:
                print(f"\nTOP-LEVEL TIMINGS:")
            else:
                print(f"\nLEVEL {level + 1} TIMINGS:")
            print("-" * 40)
            
            for full_key, total_time in by_level[level]:
                count = self.counts[full_key]
                avg_time = total_time / count if count > 0 else 0
                
                # Extract just the current level key for display
                key_parts = full_key.split('.')
                current_key = key_parts[-1]
                
                # Show parent context for deeper levels
                if level > 0:
                    parent_key = ".".join(key_parts[:-1])
                    print(f"  {current_key:20s} | {total_time:8.4f}s total | {avg_time:8.4f}s avg | {count:4d} calls | under: {parent_key}")
                else:
                    print(f"  {current_key:20s} | {total_time:8.4f}s total | {avg_time:8.4f}s avg | {count:4d} calls")
        
        print("-" * 60)
        total_time = sum(time for key, time in self.timings.items() if '.' not in key)
        print(f"TOTAL TOP-LEVEL TIME: {total_time:.4f}s")
        print("="*60)
    
    def print_summary(self, max_items=10):
        """Print a concise summary of the top time consumers"""
        if not self.timings:
            print("No timings recorded.")
            return
        
        print("\n" + "="*60)
        print("TIMING SUMMARY - Top Time Consumers")
        print("="*60)
        
        # Sort all timings by total time
        sorted_timings = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
        
        for i, (key, total_time) in enumerate(sorted_timings[:max_items]):
            count = self.counts[key]
            avg_time = total_time / count if count > 0 else 0
            # Show the full hierarchical path
            print(f"{i+1:2d}. {key:<40} | {total_time:8.4f}s total | {avg_time:8.4f}s avg | {count:4d} calls")
        
        if len(sorted_timings) > max_items:
            print(f"    ... and {len(sorted_timings) - max_items} more")
        
        print("="*60)
    
    def reset(self):
        """Reset all timings"""
        self.timings.clear()
        self.counts.clear()
        self.stack.clear()
        self.start_times.clear()
    
    def get_timing(self, key):
        """Get total time for a specific key"""
        return self.timings.get(key, 0.0)
    
    def get_breakdown(self, parent_key):
        """Get timing breakdown for all children of a parent key"""
        prefix = parent_key + "."
        children = {}
        for key, time_val in self.timings.items():
            if key.startswith(prefix):
                child_key = key[len(prefix):].split('.')[0]  # Get immediate child
                if child_key not in children:
                    children[child_key] = 0
                children[child_key] += time_val
        return children
    
    def print_breakdown(self, parent_key):
        """Print timing breakdown for a specific parent key"""
        breakdown = self.get_breakdown(parent_key)
        if not breakdown:
            print(f"No child timings found for '{parent_key}'")
            return
        
        total_parent_time = self.get_timing(parent_key)
        print(f"\nBreakdown for '{parent_key}' (total: {total_parent_time:.4f}s):")
        print("-" * 50)
        
        sorted_children = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)
        for child_key, child_time in sorted_children:
            percentage = (child_time / total_parent_time * 100) if total_parent_time > 0 else 0
            print(f"  {child_key:<20} | {child_time:8.4f}s | {percentage:5.1f}%")
    
    def get_percentage_breakdown(self, parent_key):
        """Get percentage breakdown of child timings relative to parent"""
        breakdown = self.get_breakdown(parent_key)
        total_parent_time = self.get_timing(parent_key)
        
        if total_parent_time == 0:
            return {}
        
        return {child: (time_val / total_parent_time * 100) 
                for child, time_val in breakdown.items()}
    
    def print_averaged_timings(self, num_epochs, min_time=0.001):
        """Print timings averaged per epoch (excluding first epoch setup)"""
        if not self.timings:
            print("No timings recorded.")
            return
        
        print(f"\n=== AVERAGED TIMING OVER {num_epochs} EPOCHS ===")
        print("="*80)
        
        # Build tree structure with averaged times
        tree = {}
        for key, total_time in self.timings.items():
            if total_time >= min_time:
                # For epoch-level timings, divide by number of epochs
                if 'epoch' in key and not key.endswith('epoch'):
                    # This is a sub-timing of epochs, so average it
                    avg_time = total_time / num_epochs
                else:
                    avg_time = total_time / self.counts[key] if self.counts[key] > 0 else 0
                
                if avg_time >= min_time / num_epochs:  # Adjust threshold for averaged times
                    parts = key.split('.')
                    current = tree
                    for part in parts:
                        if part not in current:
                            current[part] = {'_children': {}, '_time': 0, '_count': 0}
                        current = current[part]['_children']
                    # Store timing info at the leaf
                    leaf = tree
                    for part in parts[:-1]:
                        leaf = leaf[part]['_children']
                    leaf[parts[-1]]['_time'] = avg_time
                    leaf[parts[-1]]['_count'] = self.counts[key] // num_epochs if 'epoch' in key else self.counts[key]
        
        # Print tree recursively
        def print_node(node_dict, prefix="", is_last=True):
            items = list(node_dict.items())
            for i, (key, value) in enumerate(items):
                is_last_item = (i == len(items) - 1)
                
                # Current prefix for this line
                current_prefix = "└── " if is_last_item else "├── "
                
                # Format timing info
                time_val = value['_time']
                count_val = value['_count']
                if time_val > 0:
                    timing_info = f"{time_val:8.4f}s avg/epoch | {count_val:4d} calls/epoch"
                    print(f"{prefix}{current_prefix}{key:<25} | {timing_info}")
                else:
                    print(f"{prefix}{current_prefix}{key}")
                
                # Prepare prefix for children
                if value['_children']:
                    child_prefix = prefix + ("    " if is_last_item else "│   ")
                    print_node(value['_children'], child_prefix, is_last_item)
        
        print_node(tree)
        print("="*80)


# Global timer instance
timer = HierarchicalTimer()


trajectories = [np.load(f"../../data/boids/raw/{f}") for f in os.listdir("../../data/boids/raw") if f.endswith(".npy")]
print(len(trajectories))


def plot_state(trajectory, timestep):
    fig, ax = plt.subplots()
    # Plot dots for the boids
    ax.scatter(trajectory[timestep, :, 0], trajectory[timestep, :, 1])
    # plot the boid velocities as arrows
    for i in range(trajectory.shape[1]):
        # NOTE: The arrows are made larger for effect
        ax.arrow(trajectory[timestep, i, 0], trajectory[timestep, i, 1], trajectory[timestep, i, 2]*5, trajectory[timestep, i, 3]*5)
    return ax

# Plot timesteps 0, 250, 500, 750, 999 for the first trajectory
trajectory = trajectories[0]
fig, axs = plt.subplots(1, 5, figsize=(20, 4))
for i, t in enumerate([0, 250, 500, 750, 999]):
    axs[i].set_title(f"Timestep {t}")
    # Plot dots for the boids
    axs[i].scatter(trajectory[t, :, 0], trajectory[t, :, 1])
    # plot the boid velocities as arrows
    for j in range(trajectory.shape[1]):
        # NOTE: The arrows are made larger for effect
        axs[i].arrow(trajectory[t, j, 0], trajectory[t, j, 1], trajectory[t, j, 2]*5, trajectory[t, j, 3]*5)
plt.show()

class EGNNLayer(torch.nn.Module):
    """
    E(n) Equivariant Graph Neural Network Layer
    Based on Satorras et al. "E(n) Equivariant Graph Neural Networks"
    """
    
    def __init__(self, hidden_node_dim=64, hidden_edge_dim=32, width=1000.0, height=1000.0):
        super(EGNNLayer, self).__init__()
        
        # Periodic box sizes (for minimal image convention)
        self.width = float(width)
        self.height = float(height)
        
        # φ_e: Edge message function - takes [h_i, h_j, ||x_i - x_j||²]
        # Paper: Input → LinearLayer → Swish → LinearLayer → Swish → Output
        self.phi_e = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_node_dim + 1, hidden_edge_dim),
            torch.nn.SiLU(),  # SiLU is the same as Swish
            torch.nn.Linear(hidden_edge_dim, hidden_edge_dim),
            torch.nn.SiLU()
        )
        
        # φ_x: Coordinate weight function - takes edge message and outputs scalar
        # Paper: m_ij → LinearLayer → Swish → LinearLayer → Output
        self.phi_x = torch.nn.Sequential(
            torch.nn.Linear(hidden_edge_dim, hidden_edge_dim),
            torch.nn.SiLU(),  # SiLU is the same as Swish
            torch.nn.Linear(hidden_edge_dim, 1)
        )
        
        # φ_v: Velocity scaling function - takes node embedding and outputs velocity scale
        self.phi_v = torch.nn.Sequential(
            torch.nn.Linear(hidden_node_dim, hidden_node_dim),
            torch.nn.SiLU(),  # SiLU is the same as Swish
            torch.nn.Linear(hidden_node_dim, 2)  # Output 2D for velocity scaling
        )
        
        # φ_h: Node update function - takes [h_i, aggregate_messages]
        # Paper: [h_i, m_i] → LinearLayer → Swish → LinearLayer → Addition(h_i) → h_i^{l+1}
        # Note: We'll implement the residual connection in the forward pass
        self.phi_h = torch.nn.Sequential(
            torch.nn.Linear(hidden_node_dim + hidden_edge_dim, hidden_node_dim),
            torch.nn.SiLU(),  # SiLU is the same as Swish
            torch.nn.Linear(hidden_node_dim, hidden_node_dim)
        )
        
        # Initialize weights with smaller values
        self._init_weights()
        
    def _minimal_image(self, delta, device):
        """Apply minimal image convention to 2D displacements under PBC."""
        half_sizes = torch.tensor([self.width / 2.0, self.height / 2.0], device=device, dtype=delta.dtype)
        sizes = torch.tensor([self.width, self.height], device=device, dtype=delta.dtype)
        return torch.remainder(delta + half_sizes, sizes) - half_sizes

    def _init_weights(self):
        """Initialize weights using Xavier uniform with default gain."""
        for module in [self.phi_e, self.phi_x, self.phi_v, self.phi_h]:
            for layer in module:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)
        # Zero-init final coord weight layer so initial coordinate weights are near zero
        if isinstance(self.phi_x[-1], torch.nn.Linear):
            torch.nn.init.zeros_(self.phi_x[-1].weight)
            torch.nn.init.zeros_(self.phi_x[-1].bias)
        
    def forward(self, nodes, pos, vel_init, edge_index):
        """
        Single EGNN layer forward pass
        
        Args:
            nodes: [N, hidden_node_dim] node embeddings
            pos: [N, 2] positions
            vel_init: [N, 2] initial velocities (preserved throughout all layers)
            edge_index: [2, E] edge connectivity
            
        Returns:
            nodes_new: [N, hidden_node_dim] updated node embeddings
            pos_new: [N, 2] updated positions
            vel_new: [N, 2] updated velocities (for this layer only)
        """
        with timer.time("egnn_layer"):
            N = nodes.shape[0]
            
            # Compute messages m_ij = φ_e(h_i, h_j, ||x_i - x_j||²)
            with timer.time("messages"):
                messages = self.compute_messages(nodes, pos, edge_index)
            
            # Compute φ_v(h_i)*v_i^{init} (always use initial velocities, not updated ones)
            vel_node_update = self.phi_v(nodes) * vel_init  # [N, 2]
            
            # Compute (1/N)*Σ_{j≠i}(x_i - x_j)*φ_x(m_ij)
            with timer.time("velocity_updates"):
                vel_message_update = self.compute_velocity_message_update(pos, messages, edge_index, N)
                # Update velocity: v_i^{l+1} = φ_v(h_i)*v_i^{init} + (1/N)*Σ_{j≠i}(x_i - x_j)*φ_x(m_ij)
                vel_new = vel_node_update + vel_message_update
                # Update position: x_i^{l+1} = x_i + v_i^{l+1}
                pos_new = pos + vel_new
            
            # Aggregate messages and update node embeddings
            with timer.time("node_updates"):
                aggregate_messages = self.aggregate_messages(messages, edge_index, N)
                # Apply phi_h and add residual connection as per paper
                node_update = self.phi_h(torch.cat([nodes, aggregate_messages], dim=1))
                nodes_new = nodes + node_update  # Residual connection: h_i^{l+1} = h_i^l + phi_h(...)
            
            return nodes_new, pos_new, vel_new
    
    def compute_messages(self, nodes, pos, edge_index):
        """
        Compute messages m_ij = φ_e(h_i, h_j, ||x_i - x_j||²)
        """
        edge_source_ids, edge_target_ids = edge_index  # [E] each
        
        # Get node features for each edge
        h_i = nodes[edge_target_ids]  # [E, hidden_node_dim] - target nodes
        h_j = nodes[edge_source_ids]  # [E, hidden_node_dim] - source nodes
        
        # Get positions for each edge
        pos_i = pos[edge_target_ids]  # [E, 2] - target positions
        pos_j = pos[edge_source_ids]  # [E, 2] - source positions
        
        # Compute minimal-image position differences
        pos_diff_ij = self._minimal_image(pos_i - pos_j, pos.device)  # [E, 2]
        
        # Compute distances ||x_i - x_j||
        dist_squared = (pos_diff_ij ** 2).sum(dim=1, keepdim=True)  # [E, 1]
        dist = torch.sqrt(dist_squared)  # [E, 1]
        dist_normalized = dist / 1000.0  # [E, 1]

        # Create message input [h_i, h_j, ||x_i - x_j||²]
        message_input = torch.cat([h_i, h_j, dist_normalized], dim=1)  # [E, 2*hidden_node_dim + 1]
        # Compute messages
        messages = self.phi_e(message_input)  # [E, hidden_edge_dim]
        # Lightweight diagnostics
        try:
            self._last_messages_abs_mean = messages.detach().abs().mean()
        except Exception:
            self._last_messages_abs_mean = torch.tensor(0.0, device=messages.device)
        
        return messages
    
    def compute_velocity_message_update(self, pos, messages, edge_index, N):
        """
        Compute (1/N)*Σ_{j≠i}(x_i - x_j)*φ_x(m_ij)
        """
        edge_source_ids, edge_target_ids = edge_index  # [E] each
        
        # Get positions for each edge
        pos_i = pos[edge_target_ids]  # [E, 2] - target positions  
        pos_j = pos[edge_source_ids]  # [E, 2] - source positions
        
        # Compute minimal-image position differences (x_i - x_j)
        pos_diff = self._minimal_image(pos_i - pos_j, pos.device)  # [E, 2]
        
        # Compute coordinate weights φ_x(m_ij)
        coord_weights = self.phi_x(messages)  # [E, 1]
        
        # Diagnostics
        try:
            self._last_coord_weight_abs_mean = coord_weights.detach().abs().mean()
        except Exception:
            self._last_coord_weight_abs_mean = torch.tensor(0.0, device=pos.device)
        
        # Weighted position differences
        weighted_pos_diff = pos_diff * coord_weights  # [E, 2]
        
        # Aggregate by target nodes (sum over j for each i)
        vel_update = torch.zeros(N, 2, device=pos.device)
        vel_update.index_add_(0, edge_target_ids, weighted_pos_diff)
        # More diagnostics
        try:
            self._last_vel_update_norm_mean = vel_update.detach().norm(dim=1).mean()
        except Exception:
            self._last_vel_update_norm_mean = torch.tensor(0.0, device=pos.device)
        
        # Normalize by per-node in-degree (handles batched graphs correctly)
        degrees = torch.bincount(edge_target_ids, minlength=pos.size(0)).unsqueeze(1)
        vel_update = vel_update / degrees.clamp(min=1)
        
        return vel_update
    
    def aggregate_messages(self, messages, edge_index, N):
        """
        Aggregate messages: m_i = Σ_{j≠i} m_ij
        """
        edge_source_ids, edge_target_ids = edge_index
        
        # Aggregate messages by target nodes
        aggregate_messages = torch.zeros(N, messages.shape[1], device=messages.device)
        aggregate_messages.index_add_(0, edge_target_ids, messages)
        
        return aggregate_messages
    
class EGNN(torch.nn.Module):
    """
    Full E(n) Equivariant Graph Neural Network for Boids
    """
    
    def __init__(self, hidden_node_dim=64, hidden_edge_dim=32, num_layers=4, num_nodes=25, width=1000.0, height=1000.0):
        super(EGNN, self).__init__()
        
        # Store dims
        self.hidden_node_dim = hidden_node_dim
        
        # EGNN layers
        self.layers = torch.nn.ModuleList([
            EGNNLayer(hidden_node_dim, hidden_edge_dim, width=width, height=height) 
            for _ in range(num_layers)
        ])
        
    def forward(self, data):
        """
        Args:
            data: PyG data object with:
                - x: [N, 4] node features [pos_x, pos_y, vel_x, vel_y]
                - edge_index: [2, E] edge connectivity
        
        Returns:
            output: [N, 4] in format [pos_delta_x, pos_delta_y, vel_x_new, vel_y_new]
            
        Note: Output predicts position deltas and new velocities for balanced loss computation
        """
        with timer.time("egnn_forward"):
            # Extract features
            pos = data.x[:, :2]        # [N_total, 2] positions
            vel_init = data.x[:, 2:]   # [N_total, 2] initial velocities (preserved)

            # Initialize node embeddings simply as zeros; batching is handled implicitly by concatenation
            nodes = torch.zeros(pos.size(0), self.hidden_node_dim, device=pos.device)
            
            # Apply EGNN layers, always using the initial velocities
            self.last_layer_stats = []
            for i, layer in enumerate(self.layers):
                with timer.time(f"layer_{i}"):
                    nodes, pos, vel_current = layer(nodes, pos, vel_init, data.edge_index)
                    # Note: We get vel_current from each layer but always pass vel_init to the next layer
                    # Collect lightweight per-layer stats for visibility
                    stats = {
                        "messages_abs_mean": float(getattr(layer, "_last_messages_abs_mean", torch.tensor(0.0)).detach().item()),
                        "coord_weight_abs_mean": float(getattr(layer, "_last_coord_weight_abs_mean", torch.tensor(0.0)).detach().item()),
                        "vel_update_norm_mean": float(getattr(layer, "_last_vel_update_norm_mean", torch.tensor(0.0)).detach().item()),
                    }
                    self.last_layer_stats.append(stats)
            
            # Calculate position delta from initial positions
            pos_delta = pos - data.x[:, :2]  # [N, 2] - difference from input positions
            
            # Return position deltas and final velocities for loss computation
            return torch.cat([pos_delta, vel_current], dim=1)  # [N, 4] - [pos_delta_x, pos_delta_y, vel_x, vel_y]


class AR_EGNN_Dataset(InMemoryDataset):
    """
    Dataset for EGNN in the format:
    Input: [pos_x, pos_y, vel_x, vel_y]
    Output: [pos_delta_x, pos_delta_y, vel_x_new, vel_y_new] (position deltas and new velocities)
    
    Position deltas have similar magnitude to velocities, making loss contributions balanced.
    Node embeddings are learnable parameters in the model, not stored in the dataset.
    """
    def __init__(self, raw_data_path, processed_data_path, root=None, transform=None, pre_transform=None, post_transform=None, solution_idx_range=(0, 25), timesteps=1000, processed_file_name="AR1_EGNN_Boids.pt", hidden_node_dim=1, width=1000, height=1000):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.solution_idx_range = solution_idx_range
        self.timesteps = timesteps
        self.processed_file_name = processed_file_name
        self.hidden_node_dim = hidden_node_dim
        self.width = width
        self.height = height
        self.pre_transform = pre_transform
        self.transform = transform
        self.post_transform = post_transform
        super(AR_EGNN_Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return [self.processed_file_name]

    @property
    def raw_file_names(self):
        return [pfn for pfn in os.listdir(self.raw_data_path) if (self.solution_idx_range[0] <= int(pfn.split("_")[-1][:-4]) < self.solution_idx_range[1])]
    
    def download(self):
        pass
    
    def __len__(self):
        return (self.timesteps - 1) * (self.solution_idx_range[1] - self.solution_idx_range[0])

    def process(self):
        with timer.time("dataset_processing"):
            data_list = []
            for idx, raw_path in enumerate(self.raw_file_names):
                trajectory = np.load(self.raw_data_path + raw_path)

                if self.transform is not None:
                    trajectory = self.transform(trajectory)

                for t in range(trajectory.shape[0] - 1):
                    # Current state at time t
                    curr_state = torch.tensor(trajectory[t], dtype=torch.float)  # [N, 4] - [pos_x, pos_y, vel_x, vel_y]
                    next_state = torch.tensor(trajectory[t+1], dtype=torch.float)  # [N, 4]
                    
                    N = curr_state.shape[0]
                    
                    # Create input in format [pos_x, pos_y, vel_x, vel_y]
                    # No embeddings in dataset - they're learnable parameters in the model
                    x = curr_state  # [N, 4] - [pos_x, pos_y, vel_x, vel_y]
                    
                    # Create target as [pos_delta_x, pos_delta_y, vel_x_new, vel_y_new]
                    # Position delta with periodic boundary conditions handled properly
                    raw_pos_delta = next_state[:, :2] - curr_state[:, :2]  # [N, 2] - raw position differences
                    
                    # Wrap deltas to the minimal displacement in [-L/2, L/2)
                    # Use component-wise modulo with domain sizes (width, height)
                    half_sizes = torch.tensor([self.width / 2.0, self.height / 2.0], device=raw_pos_delta.device, dtype=raw_pos_delta.dtype)
                    sizes = torch.tensor([self.width, self.height], device=raw_pos_delta.device, dtype=raw_pos_delta.dtype)
                    pos_delta = torch.remainder(raw_pos_delta + half_sizes, sizes) - half_sizes
                    
                    vel_new = next_state[:, 2:]  # [N, 2] - new velocities
                    y = torch.cat([pos_delta, vel_new], dim=1)  # [N, 4] - [pos_delta_x, pos_delta_y, vel_x, vel_y]

                    # Fully connected graph
                    edge_index = torch.tensor([[i, j] for i in range(N) for j in range(N) if i != j], dtype=torch.long).t().contiguous()
        
                    data = Data(x=x, y=y, edge_index=edge_index)
                    if self.post_transform is not None:
                        data = self.post_transform(data)
                    data_list.append(data)
                
            data, slices = self.collate(data_list)
            os.makedirs(self.processed_data_path, exist_ok=True)
            torch.save((data, slices), self.processed_data_path + self.processed_file_name)

    def __getitem__(self, idx):
        return self.get(idx)
    
    def __repr__(self):
        return f'{self.__class__.__name__}({len(self)})'
    

class Trainer:
    def __init__(self, model, train_dataset, validation_dataset, batch_size=1, lr=0.0001, epochs=100, loss_fn=torch.nn.MSELoss(), model_name= "Model.pt", grad_clip_max_norm=1.0):
        """
        Simple Trainer class to train a PyTorch (geometric) model on a dataset.

        Args:
            model: PyTorch model to train
            train_dataset: PyTorch dataset to train on
            validation_dataset: PyTorch dataset to validate on
            batch_size: Batch size for training
            lr: Learning rate
            epochs: Number of epochs to train for
            loss_fn: Loss function to use
        """
        self.model = model
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.grad_clip_max_norm = grad_clip_max_norm

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using device:", self.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # TensorBoard writer
        os.makedirs("../../runs", exist_ok=True)
        self.writer = SummaryWriter(log_dir=f"../../runs/egnn_{int(time.time())}")

        self.train_loader = self.make_data_loader(self.train_dataset)
        self.validation_loader = self.make_data_loader(self.validation_dataset, shuffle=False)

        # Ensure model directory exists
        os.makedirs("../../models", exist_ok=True)

        # Compute constant-velocity baselines
        self._report_constant_velocity_baseline()

    def make_data_loader(self, dataset, shuffle=True):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def train_loop(self):
        """
        Train loop for the model
        """
        with timer.time("training"):
            best_model_loss = np.inf
            for epoch in range(self.epochs):
                with timer.time("epoch"):
                    # Train the model
                    with timer.time("train_phase"):
                        self.model.train()
                        sum_train_loss = 0.0
                        sum_train_grad_norm = 0.0
                        num_train_batches = 0
                        # Per-layer epoch accumulators
                        num_layers = len(self.model.layers)
                        layer_sums = [
                            {"messages_abs_mean": 0.0, "coord_weight_abs_mean": 0.0, "vel_update_norm_mean": 0.0}
                            for _ in range(num_layers)
                        ]
                        for i, data in enumerate(self.train_loader):
                            with timer.time("batch"):
                                data = data.to(self.device)
                                
                                # Forward pass
                                with timer.time("forward"):
                                    self.optimizer.zero_grad()
                                    out = self.model(data)
                                    # Balanced loss: include position deltas and velocities
                                    loss_pos = self.loss_fn(out[:, :2], data.y[:, :2])
                                    loss_vel = self.loss_fn(out[:, 2:], data.y[:, 2:])
                                    loss = loss_pos + loss_vel
                                
                                # Debug first few iterations to understand what's happening
                                if epoch == 0 and i < 3:
                                    print(f"Batch {i}:")
                                    print(f"  Input range positions: [{data.x[:, :2].min():.4f}, {data.x[:, :2].max():.4f}]")
                                    print(f"  Input range velocities: [{data.x[:, 2:].min():.4f}, {data.x[:, 2:].max():.4f}]")
                                    print(f"  Output range positions: [{out[:, :2].min():.4f}, {out[:, :2].max():.4f}]")
                                    print(f"  Output range velocities: [{out[:, 2:].min():.4f}, {out[:, 2:].max():.4f}]")
                                    print(f"  Target range positions: [{data.y[:, :2].min():.4f}, {data.y[:, :2].max():.4f}]")
                                    print(f"  Target range velocities: [{data.y[:, 2:].min():.4f}, {data.y[:, 2:].max():.4f}]")
                                    print(f"  Loss: {loss.item():.6f}")
                                
                                # Check for NaN loss
                                if torch.isnan(loss):
                                    print(f"NaN loss detected at epoch {epoch}, batch {i}")
                                    return
                                
                                # Backward pass
                                with timer.time("backward"):
                                    loss.backward()
                                    # Gradient clipping and norm logging
                                    total_norm = float(clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm))
                                    sum_train_grad_norm += total_norm
                                    self.optimizer.step()
                                
                                sum_train_loss += float(loss.item())
                                # Accumulate per-layer stats
                                for li, stats in enumerate(getattr(self.model, "last_layer_stats", [])):
                                    for k in layer_sums[li].keys():
                                        layer_sums[li][k] += float(stats.get(k, 0.0))
                                num_train_batches += 1
                        mean_train_loss = sum_train_loss / max(1, num_train_batches)
                        mean_train_grad_norm = sum_train_grad_norm / max(1, num_train_batches)
                    
                    # Validate the model
                    with timer.time("validation"):
                        self.model.eval()
                        sum_val_loss = 0.0
                        num_val_batches = 0
                        # Per-layer epoch accumulators (validation)
                        num_layers = len(self.model.layers)
                        layer_sums_val = [
                            {"messages_abs_mean": 0.0, "coord_weight_abs_mean": 0.0, "vel_update_norm_mean": 0.0}
                            for _ in range(num_layers)
                        ]
                        with torch.no_grad():
                            for i, data in enumerate(self.validation_loader):
                                data = data.to(self.device)
                                out = self.model(data)
                                # Balanced loss: include position deltas and velocities
                                loss_pos = self.loss_fn(out[:, :2], data.y[:, :2])
                                loss_vel = self.loss_fn(out[:, 2:], data.y[:, 2:])
                                loss = loss_pos + loss_vel
                                sum_val_loss += float(loss.item())
                                # Accumulate per-layer stats
                                for li, stats in enumerate(getattr(self.model, "last_layer_stats", [])):
                                    for k in layer_sums_val[li].keys():
                                        layer_sums_val[li][k] += float(stats.get(k, 0.0))
                                num_val_batches += 1
                            mean_val_loss = sum_val_loss / max(1, num_val_batches)

                    # Save best model
                    if mean_val_loss < best_model_loss:
                        best_model_loss = mean_val_loss
                        torch.save(self.model.state_dict(), f"../../models/{self.model_name}")
                    
                    print(f"Epoch {epoch}, Train Loss: {mean_train_loss:.6f}, Val Loss: {mean_val_loss:.6f}")

                    # LR logging (no scheduler)
                    current_lr = self.optimizer.param_groups[0]['lr']
                    # TensorBoard logging (epoch-level)
                    self.writer.add_scalar("loss/train", mean_train_loss, epoch)
                    self.writer.add_scalar("loss/val", mean_val_loss, epoch)
                    self.writer.add_scalar("optimizer/lr", current_lr, epoch)
                    self.writer.add_scalar("grad/total_norm", mean_train_grad_norm, epoch)
                    # Log averaged per-layer stats
                    if num_train_batches > 0:
                        for li in range(len(layer_sums)):
                            for k, v in layer_sums[li].items():
                                self.writer.add_scalar(f"train/layer_{li}/{k}", v / num_train_batches, epoch)
                    if num_val_batches > 0:
                        for li in range(len(layer_sums_val)):
                            for k, v in layer_sums_val[li].items():
                                self.writer.add_scalar(f"val/layer_{li}/{k}", v / num_val_batches, epoch)
                
                # Print timing report after each epoch
                if epoch == 0:  # Print detailed timing for first epoch
                    print(f"\n=== EPOCH {epoch} TIMING ===")
                    timer.print_timings()
                elif epoch % 10 == 0 and epoch > 0:  # Print timing every 10 epochs
                    print(f"\n=== EPOCH {epoch} TIMING ===")
                    timer.print_timings()

        # Close writer at end
        self.writer.close()

    def _report_constant_velocity_baseline(self):
        """Compute constant-velocity baseline with current training objective (pos_delta + vel)."""
        def dataset_combined_loss(dataset):
            loss_sum = 0.0
            count = 0
            for i in range(len(dataset)):
                d = dataset[i]
                # Constant-velocity predictor: Δx ≈ v_t, v_{t+1} ≈ v_t
                pred_pos_delta = d.x[:, 2:]
                pred_vel = d.x[:, 2:]
                target_pos_delta = d.y[:, :2]
                target_vel = d.y[:, 2:]
                loss = torch.mean((pred_pos_delta - target_pos_delta) ** 2) + \
                       torch.mean((pred_vel - target_vel) ** 2)
                loss_sum += float(loss.item())
                count += 1
            return loss_sum / max(1, count)
        train_loss = dataset_combined_loss(self.train_dataset)
        val_loss = dataset_combined_loss(self.validation_dataset)
        print(f"Baseline (const vel) combined loss -> Train: {train_loss:.6f}, Val: {val_loss:.6f}")


# Train the EGNN model and perform rollouts
print("=== Training EGNN Model ===")

# Create and train EGNN model
egnn_model = EGNN(hidden_node_dim=64, hidden_edge_dim=128, num_layers=4, num_nodes=25)

# Create datasets
train_dataset = AR_EGNN_Dataset(
    raw_data_path="../../data/boids/raw/", 
    processed_data_path="../../data/boids/processed/", 
    root="../../data/boids/", 
    solution_idx_range=(0, 15), 
    timesteps=1000, 
    processed_file_name="AR1_EGNN_Train.pt"
)
validation_dataset = AR_EGNN_Dataset(
    raw_data_path="../../data/boids/raw/", 
    processed_data_path="../../data/boids/processed/", 
    root="../../data/boids/", 
    solution_idx_range=(16, 25), 
    timesteps=1000, 
    processed_file_name="AR1_EGNN_Val.pt"
)

egnn_trainer = Trainer(
    model=egnn_model, 
    train_dataset=train_dataset,
    validation_dataset=validation_dataset,
    batch_size=16, 
    lr=0.0001,
    epochs=20, 
    loss_fn=torch.nn.MSELoss(), 
    model_name="EGNN-Model.pt"
)

print("Starting EGNN training...")
egnn_trainer.train_loop()

# Print final timing report
timer.print_averaged_timings(egnn_trainer.epochs)
print("Training completed.")

print("\n=== Generating EGNN Rollouts ===")

# Load the best trained model
egnn_model = EGNN(hidden_node_dim=64, hidden_edge_dim=128, num_layers=4, num_nodes=25)
egnn_model.load_state_dict(torch.load("../../models/EGNN-Model.pt"))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
egnn_model.to(device)
egnn_model.eval()

def compute_egnn_rollouts(model, dataset, timesteps=1000, device='cuda', width=1000, height=1000):
    """
    Predict the rollouts of the EGNN model on the dataset
    
    Args:
        model: EGNN model (outputs position deltas and velocities)
        dataset: Dataset with initial states in format [pos_x, pos_y, vel_x, vel_y]
        timesteps: Number of timesteps to predict
        device: Device to run the model on  
        width: Width of the PBC box
        height: Height of the PBC box
    Returns:
        rollouts: Rollouts of the model on the dataset
        - Should be a torch tensor of shape (Batch, Timesteps, Boids, 4) in format [pos_x, pos_y, vel_x, vel_y]
    """
    with timer.time("rollout_generation"):
        # Output rollouts in the standard format [pos_x, pos_y, vel_x, vel_y] for compatibility with analysis functions
        rollouts = torch.empty((len(dataset), timesteps, dataset[0].x.shape[0], 4), device=device)

        model.eval()
        with torch.no_grad():
            for idx, init_data in enumerate(dataset):
                # Get the initial state
                model_data = init_data.clone().to(device)
                current_pos = model_data.x[:, :2].clone()  # Track absolute positions

                for t in range(timesteps):
                    with timer.time("rollout_step"):
                        # Forward pass: input [pos_x, pos_y, vel_x, vel_y] 
                        # -> output [pos_delta_x, pos_delta_y, vel_x_new, vel_y_new]
                        out_data = model(model_data)
                        
                        # Convert position deltas to absolute positions
                        pos_deltas = out_data[:, :2]  # [N, 2] position deltas
                        new_velocities = out_data[:, 2:]  # [N, 2] new velocities
                        new_positions = current_pos + pos_deltas  # [N, 2] absolute positions
                        
                        # Apply periodic boundary conditions to positions
                        new_positions[:, 0] = new_positions[:, 0] % width
                        new_positions[:, 1] = new_positions[:, 1] % height
                        
                        # Create full state for storage and next iteration
                        full_state = torch.cat([new_positions, new_velocities], dim=1)  # [N, 4]
                        
                        # Store rollout in standard format [pos_x, pos_y, vel_x, vel_y]
                        rollouts[idx, t] = full_state.clone().cpu()
                        
                        # Update for next timestep
                        current_pos = new_positions.clone()
                        model_data.x = full_state
                
        return rollouts

# Load dataset for rollout
def keep_01(data):
    return data[0:2, :, :]

initial_states_validation_dataset = AR_EGNN_Dataset(
    raw_data_path="../../data/boids/raw/", 
    processed_data_path="../../data/boids/processed/", 
    root="../../data/boids/", 
    solution_idx_range=(16, 25), 
    timesteps=2, 
    processed_file_name="AR1_VAL_init.pt",
    transform=keep_01
)

print("Generating EGNN rollouts...")
egnn_model_rollout = compute_egnn_rollouts(
    egnn_model, 
    initial_states_validation_dataset,
    timesteps=1000, 
    device=device
)

print(f"✓ EGNN rollouts generated! Shape: {egnn_model_rollout.shape}")
print("Rollout format: [pos_x, pos_y, vel_x, vel_y] for compatibility with analysis functions")

# Quick visualization test
print("\n=== Testing EGNN Animation ===")
from IPython.display import Image

from matplotlib import animation

def animate_rollout(rollouts, output_path="output/rollout.gif", width = 1000, height = 1000, max_timesteps=100):
    # rollouts of shape (Timesteps, Boids, Node_dim)
    
    # Create output directory if it does not exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Parse rollouts
    timesteps, num_boids, node_dim = rollouts.shape
    rollouts = rollouts.cpu().numpy()

    # Initialize the figure and axis
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)

    quiv = ax.quiver(rollouts[0, :, 0], rollouts[0, :, 1], rollouts[0, :, 2], rollouts[0, :, 3])
    scat = ax.scatter(rollouts[0, :, 0], rollouts[0, :, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal', adjustable='box')

    def update(frame):
        quiv.set_offsets(rollouts[frame, :, :2])
        quiv.set_UVC(rollouts[frame, :, 2], rollouts[frame, :, 3])
        scat.set_offsets(rollouts[frame, :, :2])
        return scat, quiv

    ani = animation.FuncAnimation(fig, update, frames=min(timesteps, max_timesteps), blit=False, interval=150)
    ani.save(output_path, writer="ffmpeg")
    plt.close()

animate_rollout(egnn_model_rollout[0], output_path="output/egnn_rollout.gif")
print("EGNN animation saved to output/egnn_rollout.gif")
Image(filename="output/egnn_rollout.gif")