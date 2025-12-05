import numpy as np
import torch
import hashlib
from copy import deepcopy
from env.circuit_env import CircuitEnv

class CircuitEnvWrapper:
    """
    Wrapper for CircuitEnv to make it compatible with AlphaZero/MCTS.
    Provides state hashing, cloning, and flattened action space handling.
    """
    def __init__(self, env: CircuitEnv):
        self.env = env
        self.max_nodes = env.max_nodes
        self.max_components = env.max_components
        
        # Flattened action space logic
        # MultiDiscrete([2, 3, max_nodes, max_nodes]) -> Flattened Index
        # 3 represents the number of component types (V, L, S)
        self.dims = [2, 3, self.max_nodes, self.max_nodes]
        self.action_size = np.prod(self.dims)

    def reset(self):
        return self.env.reset()

    def clone(self):
        """Deep copies the environment state."""
        new_env = CircuitEnv(self.env.initial_inventory, self.env.max_nodes)
        
        # Copy scalar state
        new_env.node_counter = self.env.node_counter
        new_env.step_count = self.env.step_count
        new_env.max_steps = self.env.max_steps
        new_env.step_count = self.env.step_count
        new_env.max_steps = self.env.max_steps
        new_env.last_cycle_count = self.env.last_cycle_count
        new_env.terminated = self.env.terminated
        
        # Deep copy mutable state
        new_env.available_components = self.env.available_components.copy()
        new_env.circuit_graph = deepcopy(self.env.circuit_graph)
        
        return CircuitEnvWrapper(new_env)

    def action_space_size(self):
        return self.action_size

    def canonical_string(self):
        """
        Generates a unique string representation of the state for MCTS hashing.
        Combines Adjacency Matrix, Inventory Mask, and Step Count.
        """
        obs = self.env._get_obs()
        adj = obs['adjacency']
        inv = obs['inventory_mask']
        
        state_bytes = adj.tobytes() + inv.tobytes() + str(self.env.step_count).encode()
        return hashlib.md5(state_bytes).hexdigest()

    def is_terminal(self):
        """
        Checks if the episode has ended and returns the final reward.
        Returns: (terminated, normalized_score)
        """
        terminated = self.env.terminated or (np.sum(self.env.available_components) == 0) or (self.env.step_count >= self.env.max_steps)
        
        if terminated:
            if hasattr(self.env, '_calculate_circuit_score'):
                score = self.env._calculate_circuit_score()
            else:
                score = 0.0 
                
            # Normalize score to [-1, 1] range (assuming max raw score is around 200)
            return True, score / 200.0
            
        return False, 0

    def step_flat(self, action_idx):
        """
        Executes a flattened action index in the environment.
        """
        # Decode flattened index to MultiDiscrete action
        action = []
        rem = action_idx
        for dim in reversed(self.dims):
            action.append(rem % dim)
            rem //= dim
        action = list(reversed(action))
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # In AlphaZero, we typically ignore intermediate rewards and focus on the final outcome
        return 0.0

    def get_action_mask(self):
        """
        Returns a flattened boolean mask of valid actions.
        """
        masks = self.env.action_masks()
        
        # Parse the concatenated masks from the environment
        idx = 0
        m_type = masks[idx : idx+2]; idx += 2
        
        # Component Type Mask (Size 3: V, L, S)
        num_types = 3
        m_comp = masks[idx : idx+num_types]; idx += num_types
        
        m_n1 = masks[idx : idx+self.max_nodes]; idx += self.max_nodes
        m_n2 = masks[idx : idx+self.max_nodes]; idx += self.max_nodes
        
        m_type = np.array(m_type, dtype=bool)
        m_comp = np.array(m_comp, dtype=bool)
        m_n1 = np.array(m_n1, dtype=bool)
        m_n2 = np.array(m_n2, dtype=bool)
        
        # Compute the Cartesian product of masks using broadcasting
        # Shape: (2, C, N, N)
        full_mask = (m_type[:, None, None, None] & 
                     m_comp[None, :, None, None] & 
                     m_n1[None, None, :, None] & 
                     m_n2[None, None, None, :])
                     
        # Explicitly forbid self-loops (n1 == n2)
        N = self.max_nodes
        for i in range(N):
            full_mask[:, :, i, i] = False
            
        return full_mask.flatten()

    def get_obs_tensor(self):
        """
        Converts the observation dictionary into a PyTorch tensor.
        Output Shape: (Channels, Height, Width)
        """
        obs = self.env._get_obs()
        adj = obs['adjacency']
        inv = obs['inventory_mask']
        
        num_types = 9
        N = self.max_nodes
        
        # 1. One-Hot Adjacency Matrix
        adj_onehot = np.zeros((num_types, N, N), dtype=np.float32)
        for t in range(num_types):
            adj_onehot[t] = (adj == t).astype(np.float32)
            
        # 2. Inventory Mask (Broadcasted)
        inv_channels = np.zeros((self.max_components, N, N), dtype=np.float32)
        for i in range(self.max_components):
            inv_channels[i, :, :] = inv[i]
            
        # 3. Node Features (Broadcasted)
        node_feats = obs['node_features']
        num_node_feats = 2
        node_feat_channels = np.zeros((num_node_feats * 2, N, N), dtype=np.float32)
        
        for f in range(num_node_feats):
            feat_col = node_feats[:, f]
            # Row broadcast
            node_feat_channels[2*f] = feat_col.reshape(-1, 1)
            # Column broadcast
            node_feat_channels[2*f+1] = feat_col.reshape(1, -1)
            
        # Concatenate all channels
        feature_map = np.concatenate([adj_onehot, inv_channels, node_feat_channels], axis=0)
        
        return torch.FloatTensor(feature_map)
