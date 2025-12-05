import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AlphaZeroNet(nn.Module):
    """
    AlphaZero Dual-Headed Neural Network.
    Predicts both the policy (action probabilities) and the value (state evaluation).
    """
    def __init__(self, input_shape, num_actions, num_res_blocks=4, num_filters=64):
        """
        Args:
            input_shape: (Channels, Height, Width) of the input observation.
            num_actions: Total number of possible actions (flattened).
            num_res_blocks: Number of residual blocks in the backbone.
            num_filters: Number of convolutional filters.
        """
        super(AlphaZeroNet, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        # 1. Convolutional Backbone
        self.conv_input = nn.Sequential(
            nn.Conv2d(input_shape[0], num_filters, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, num_filters), # Replaced InstanceNorm with LayerNorm (GroupNorm(1)) for DirectML stability
            nn.ReLU()
        )
        
        # Residual Tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])
        
        # 2. Policy Head
        # Outputs logits for action probabilities P(a|s)
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1, stride=1),
            nn.GroupNorm(1, 2), # LayerNorm
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * input_shape[1] * input_shape[2], num_actions)
        )
        
        # 3. Value Head
        # Outputs scalar value V(s) in range [-1, 1]
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1, stride=1),
            nn.GroupNorm(1, 1), # LayerNorm
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * input_shape[1] * input_shape[2], 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_input(x)
        
        for block in self.res_blocks:
            x = block(x)
            
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value

class ResidualBlock(nn.Module):
    """
    Standard Residual Block with 2 Convolutional layers and Skip Connection.
    """
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.GroupNorm(1, num_filters) # Replaced InstanceNorm with LayerNorm
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.GroupNorm(1, num_filters) # Replaced InstanceNorm with LayerNorm
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
