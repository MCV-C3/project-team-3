import torch
import torch.nn as nn
from typing import List


class SimpleModel(nn.Module):
    """
    Simple MLP for patch-based image classification
    
    Architecture:
        Input -> layer1 (hidden_d) -> ReLU -> layer2 (hidden_d) -> ReLU -> output_layer (output_d)
    """
    
    def __init__(self, input_d: int, hidden_d: int, output_d: int):
        """
        Args:
            input_d: Input dimension (C * patch_size * patch_size)
            hidden_d: Hidden layer dimension
            output_d: Output dimension (number of classes)
        """
        super(SimpleModel, self).__init__()

        self.input_d = input_d
        self.hidden_d = hidden_d
        self.output_d = output_d

        self.layer1 = nn.Linear(input_d, hidden_d)
        self.layer2 = nn.Linear(hidden_d, hidden_d)
        self.output_layer = nn.Linear(hidden_d, output_d)

        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [batch_size, input_d] or [batch_size, C, H, W]
            
        Returns:
            Output tensor of shape [batch_size, output_d]
        """
        x = x.view(x.shape[0], -1)
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.output_layer(x)
        
        return x

# -------------------------
# Patch MLP (para variar layers/neurons)
# -------------------------
class PatchMlp(nn.Module):
    def __init__(self, input_d: int, hidden_dims: List[int], output_d: int):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_d
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_d))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.net(x)
