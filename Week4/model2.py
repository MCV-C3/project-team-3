import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import *
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

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

def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    B, C, H, W = x.size()
    assert C % groups == 0, "Channels must be divisible by groups"

    x = x.view(B, groups, C // groups, H, W)
    x = x.transpose(1, 2).contiguous()
    x = x.view(B, C, H, W)
    return x


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


class DepthwiseConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float):
        super().__init__()

        # 1) Depth-wise 3x3 convolution
        self.depthwise = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch,
            bias=False
        )

        # 2) Point-wise 1x1 convolution
        self.pointwise = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class FireBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        squeeze_ch: int,
        expand_ch: int,
        dropout: float
    ):
        super().__init__()

        # Squeeze layer (1x1)
        self.squeeze = nn.Conv2d(
            in_ch,
            squeeze_ch,
            kernel_size=1,
            bias=False
        )
        self.squeeze_act = nn.ReLU(inplace=True)

        # Expand layers
        self.expand_1x1 = nn.Conv2d(
            squeeze_ch,
            expand_ch,
            kernel_size=1,
            bias=False
        )

        self.expand_3x3 = nn.Conv2d(
            squeeze_ch,
            expand_ch,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.expand_act = nn.ReLU(inplace=True)

        self.norm = nn.BatchNorm2d(2 * expand_ch)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze(x)
        x = self.squeeze_act(x)

        x1 = self.expand_1x1(x)
        x3 = self.expand_3x3(x)

        x = torch.cat([x1, x3], dim=1)  # channel concat
        x = self.expand_act(x)
        x = self.norm(x)
        x = self.pool(x)

        return x
    

class ShuffleBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        groups: int,
        dropout: float
    ):
        super().__init__()

        assert in_ch % groups == 0
        assert out_ch % groups == 0

        mid_ch = out_ch // 4  # bottleneck (standard ShuffleNet choice)

        # 1x1 grouped conv (reduce)
        self.conv1 = nn.Conv2d(
            in_ch,
            mid_ch,
            kernel_size=1,
            groups=groups,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(mid_ch)

        # 3x3 depth-wise conv
        self.dwconv = nn.Conv2d(
            mid_ch,
            mid_ch,
            kernel_size=3,
            padding=1,
            groups=mid_ch,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_ch)

        # 1x1 grouped conv (expand)
        self.conv3 = nn.Conv2d(
            mid_ch,
            out_ch,
            kernel_size=1,
            groups=groups,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = channel_shuffle(x, self.groups)

        x = self.dwconv(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)

        x = self.pool(x)
        return x




@register_model('baseline_dw')
class BaselineDepthwise(BaseModel):
    def __init__(self, num_classes: int = 8, depth: int = 5, dropout: float = 0.2):
        super().__init__()

        block_list = []
        output_channels = 3

        for i in range(depth):
            input_channels = output_channels
            output_channels = 2 ** ((i // 2) + 5)

            block_list.append(
                DepthwiseConvBlock(
                    input_channels,
                    output_channels,
                    dropout
                )
            )

        self.block_list = nn.ModuleList(block_list)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.block_list:
            x = block(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits
    

@register_model('baseline_fire')
class BaselineFire(BaseModel):
    def __init__(
        self,
        num_classes: int = 8,
        depth: int = 5,
        dropout: float = 0.2,
        squeeze_ratio: float = 0.25
    ):
        super().__init__()

        block_list = []
        output_channels = 3  # input RGB

        for i in range(depth):
            input_channels = output_channels

            # match your original channel growth
            output_channels = 2 ** ((i // 2) + 5)

            squeeze_channels = max(1, int(output_channels * squeeze_ratio))
            expand_channels = output_channels // 2  # because concat doubles it

            block_list.append(
                FireBlock(
                    in_ch=input_channels,
                    squeeze_ch=squeeze_channels,
                    expand_ch=expand_channels,
                    dropout=dropout
                )
            )

        self.block_list = nn.ModuleList(block_list)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.block_list:
            x = block(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits
    

@register_model('baseline_shuffle')
class BaselineShuffle(BaseModel):
    def __init__(
        self,
        num_classes: int = 8,
        depth: int = 5,
        dropout: float = 0.2,
        groups: int = 4
    ):
        super().__init__()

        block_list = []
        output_channels = 3

        for i in range(depth):
            input_channels = output_channels
            output_channels = 2 ** ((i // 2) + 5)

            # Ensure divisibility
            g = min(groups, input_channels, output_channels)
            while input_channels % g != 0 or output_channels % g != 0:
                g -= 1

            block_list.append(
                ShuffleBlock(
                    in_ch=input_channels,
                    out_ch=output_channels,
                    groups=g,
                    dropout=dropout
                )
            )

        self.block_list = nn.ModuleList(block_list)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.block_list:
            x = block(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


@register_model('baseline_fire_dw')
class BaselineFireDepthwise(BaseModel):
    def __init__(self, num_classes: int = 8, depth: int = 6, dropout: float = 0.2, squeeze_ratio: float = 0.25):
        super().__init__()

        block_list = []
        output_channels = 3

        for i in range(depth):
            input_channels = output_channels
            output_channels = 2 ** ((i // 2) + 5)

            # Use Fire blocks in first half, Depthwise in second half
            if i < depth // 2:
                squeeze_channels = max(1, int(output_channels * squeeze_ratio))
                expand_channels = output_channels // 2
                block_list.append(
                    FireBlock(
                        in_ch=input_channels,
                        squeeze_ch=squeeze_channels,
                        expand_ch=expand_channels,
                        dropout=dropout
                    )
                )
            else:
                block_list.append(
                    DepthwiseConvBlock(
                        in_ch=input_channels,
                        out_ch=output_channels,
                        dropout=dropout
                    )
                )

        self.block_list = nn.ModuleList(block_list)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.block_list:
            x = block(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


@register_model('baseline_shuffle_fire')
class BaselineShuffleFire(BaseModel):
    def __init__(self, num_classes: int = 8, depth: int = 6, dropout: float = 0.2, groups: int = 4, squeeze_ratio: float = 0.25):
        super().__init__()

        block_list = []
        output_channels = 3

        for i in range(depth):
            input_channels = output_channels
            output_channels = 2 ** ((i // 2) + 5)

            # Shuffle first half, Fire second half
            if i < depth // 2:
                g = min(groups, input_channels, output_channels)
                while input_channels % g != 0 or output_channels % g != 0:
                    g -= 1
                block_list.append(
                    ShuffleBlock(
                        in_ch=input_channels,
                        out_ch=output_channels,
                        groups=g,
                        dropout=dropout
                    )
                )
            else:
                squeeze_channels = max(1, int(output_channels * squeeze_ratio))
                expand_channels = output_channels // 2
                block_list.append(
                    FireBlock(
                        in_ch=input_channels,
                        squeeze_ch=squeeze_channels,
                        expand_ch=expand_channels,
                        dropout=dropout
                    )
                )

        self.block_list = nn.ModuleList(block_list)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.block_list:
            x = block(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


@register_model('baseline_fire_shuffle_dw')
class BaselineFireShuffleDepthwise(BaseModel):
    def __init__(self, num_classes: int = 8, depth: int = 6, dropout: float = 0.2, groups: int = 4, squeeze_ratio: float = 0.25):
        super().__init__()
        block_list = []
        output_channels = 3

        for i in range(depth):
            input_channels = output_channels
            output_channels = 2 ** ((i // 2) + 5)

            if i == 0:
                # Fire first
                squeeze_channels = max(1, int(output_channels * squeeze_ratio))
                expand_channels = output_channels // 2
                block_list.append(FireBlock(input_channels, squeeze_channels, expand_channels, dropout))
            elif i < depth - 1:
                # Shuffle middle
                g = min(groups, input_channels, output_channels)
                while input_channels % g != 0 or output_channels % g != 0:
                    g -= 1
                block_list.append(ShuffleBlock(input_channels, output_channels, groups=g, dropout=dropout))
            else:
                # Depthwise last
                block_list.append(DepthwiseConvBlock(input_channels, output_channels, dropout))

        self.block_list = nn.ModuleList(block_list)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(output_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.block_list:
            x = block(x)
        x = self.gap(x)
        x = torch.flatten(x,1)
        return self.classifier(x)



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


@register_model('teacher_model')
class TeacherModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        if pretrained:
            # Load pretrained InceptionV3 model
            self.backbone = models.inception_v3(
                weights="IMAGENET1K_V1",
                aux_logits=True
            )

            # Disable auxiliary classifier safely
            self.backbone.AuxLogits = None
            self.set_parameter_requires_grad(feature_extracting=pretrained)

        else:
            self.backbone = models.inception_v3(
                weights=None,
                aux_logits=True
            )
            self.backbone.AuxLogits = None


        hidden_dim = 512
        use_batchnorm = False
        dropout = 0.5
        # ----- Classifier head -----
        layers = []

        layers.append(nn.Linear(self.backbone.fc.in_features, hidden_dim))

        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))

        layers.append(nn.ReLU())

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, num_classes))
        
        self.backbone.fc = nn.Sequential(*layers)


    def forward(self, x):
        outputs = self.backbone(x)

        # In training mode, Inception returns InceptionOutputs
        if isinstance(outputs, tuple) or hasattr(outputs, "logits"):
            return outputs.logits

        return outputs
    

    def extract_feature_maps(self, input_image:torch.Tensor):
        conv_weights =[]
        conv_layers = []
        total_conv_layers = 0

        for module in self.backbone.features.children():
            if isinstance(module, nn.Conv2d):
                total_conv_layers += 1
                conv_weights.append(module.weight)
                conv_layers.append(module)

        print("TOTAL CONV LAYERS: ", total_conv_layers)
        feature_maps = []  # List to store feature maps
        layer_names = []  # List to store layer names
        x= torch.clone(input=input_image)
        for layer in conv_layers:
            x = layer(x)
            feature_maps.append(x)
            layer_names.append(str(layer))

        return feature_maps, layer_names
        

    def extract_features_from_hooks(self, x, layers: List[str]):
        """
        Extract feature maps from specified layers.
        Args:
            x (torch.Tensor): Input tensor.
            layers (List[str]): List of layer names to extract features from.
        Returns:
            Dict[str, torch.Tensor]: Feature maps from the specified layers.
        """
        outputs = {}
        hooks = []

        def get_activation(name):
            def hook(model, input, output):
                outputs[name] = output
            return hook

        # Register hooks for specified layers
        #for layer_name in layers:
        dict_named_children = {}
        for name, layer in self.backbone.named_children():
            for n, specific_layer in layer.named_children():
                dict_named_children[f"{name}.{n}"] = specific_layer

        for layer_name in layers:
            layer = dict_named_children[layer_name]
            hooks.append(layer.register_forward_hook(get_activation(layer_name)))

        # Perform forward pass
        _ = self.forward(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return outputs


    def modify_layers(self, modify_fn: Callable[[nn.Module], nn.Module]):
        """
        Modify layers of the model using a provided function.
        Args:
            modify_fn (Callable[[nn.Module], nn.Module]): Function to modify a layer.
        """
        self.vgg16 = modify_fn(self.vgg16)


    def set_parameter_requires_grad(self, feature_extracting):
        """
        Set parameters gradients to false in order not to optimize them in the training process.
        """
        if feature_extracting:
            for param in self.backbone.parameters():
                param.requires_grad = False


    def extract_grad_cam(self, input_image: torch.Tensor, 
                         target_layer: List[Type[nn.Module]], 
                         targets: List[Type[ClassifierOutputTarget]]) -> Type[GradCAMPlusPlus]:
        with GradCAMPlusPlus(model=self.backbone, target_layers=target_layer) as cam:

            grayscale_cam = cam(input_tensor=input_image, targets=targets)[0, :]

        return grayscale_cam

    
    def freeze_all_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False


    def unfreeze_blocks(self, block_names: List[str]):
        """
        Unfreeze specific Inception blocks by name
        Example block_names: ["Mixed_7c", "Mixed_7b"]
        """
        for name, module in self.backbone.named_children():
            if name in block_names:
                for p in module.parameters():
                    p.requires_grad = True

    

