import torch.nn as nn
import torch

from typing import *

class SimpleModel(nn.Module):

    def __init__(self, input_d: int, hidden_d: int, output_d: int):
        super(SimpleModel, self).__init__()

        self.input_d = input_d
        self.hidden_d = hidden_d
        self.output_d = output_d

        self.layer1 = nn.Linear(input_d, hidden_d)
        self.layer2 = nn.Linear(hidden_d, hidden_d)
        self.output_layer = nn.Linear(hidden_d, output_d)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)

        x = self.output_layer(x)
        
        return x

class FlexibleMlp(nn.Module):
    def __init__(self, input_d: int, hidden_dims: List[int], output_d: int):
        super().__init__()
        layers = []
        input_dim = input_d
        activation = nn.ReLU()
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(activation)
            input_dim = dim
        self.last_layer = nn.Linear(input_dim, output_d)
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        if x.ndim > 2:
            x = x.view(B, -1)
        x1 = self.net(x)
        output = self.last_layer(x1)
        return output
    
    def features(self, x: torch.Tensor):
        B, C, H, W = x.shape
        if x.ndim > 2:
            x = x.view(B, -1)
        output = self.net(x)
        return output

class PatchMlp(nn.Module):
    def __init__(self, num_channels: int, hidden_dims: List[int], output_d: int, patch_size: int, aggregation_method: str):
        super().__init__()
        layers = []
        input_dim = num_channels * patch_size * patch_size
        activation = nn.ReLU()
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(activation)
            input_dim = dim
        self.last_layer = nn.Linear(input_dim, output_d)
        self.net = nn.Sequential(*layers)
        self.patch_size = patch_size
        self.aggregation_method = aggregation_method
    
    def patch_image(self, img: torch.Tensor):
        B, C, H, W = img.shape
        vertical_patches = H // self.patch_size
        horizontal_patches = W // self.patch_size
        patched_image = img.contiguous().view(-1, C, vertical_patches, self.patch_size, horizontal_patches, self.patch_size).permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C * self.patch_size * self.patch_size)
        return patched_image

    def aggregate_result(self, x: torch.Tensor, batch_size: int):
        B, L = x.shape
        x1 = x.contiguous().view(batch_size, -1, L).contiguous()

        # Aggregation
        if self.aggregation_method == 'max':
            output = x1.max(dim=1, keepdim=False).values
        elif self.aggregation_method == 'mean':
            output = x1.mean(dim=1, keepdim=False)
        return output

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        patched_x = self.patch_image(x)
        x1 = self.net(patched_x)
        output = self.last_layer(x1)
        output = self.aggregate_result(output, B)
        return output
    
    def features(self, x: torch.Tensor):
        B, C, H, W = x.shape
        if x.ndim > 2:
            x = x.view(B, -1)
        output = self.net(x)
        output = self.aggregate_result(output, B)
        return output
