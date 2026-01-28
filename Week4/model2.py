import torch
import torch.nn as nn
import torch.nn.functional as F

_MODEL_REGISTRY = {}


def register_model(name):
    def decorator(cls):
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model {name} is duplicated!")
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def build_model(model_name, **kwargs):
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(_MODEL_REGISTRY.keys())}")
    
    model_class = _MODEL_REGISTRY[model_name]
    return model_class(**kwargs)


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def get_num_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@register_model('dummy_test')
class DummyTest(BaseModel):
    def __init__(self):
        pass


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)

        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pool(x)
        return x


@register_model('baseline')
class Baseline(BaseModel):
    def __init__(self, num_classes: int = 8, depth: int = 5, dropout: float = 0.2):
        super().__init__()
        block_list = []
        output_channels = 3
        for i in range(depth):
            input_channels = output_channels
            output_channels = 2 ** ((i//2) + 5)

            block_list.append(ConvBlock(input_channels, output_channels, dropout))
        self.block_list = nn.ModuleList(block_list)

        self.gap = nn.AdaptiveAvgPool2d((1, 1)) # Global Avg Pool -> (B, output_channels, 1, 1)
        self.classifier = nn.Linear(output_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.block_list:
            x = block(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)  # (B, N)
        logits = self.classifier(x)  # (B, NumClasses)
        return logits
