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


class ResidualAdd(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x, out):
        if self.proj is not None:
            x = self.proj(x)
        return x + out


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout, use_skip=False):
        super().__init__()
        self.use_skip = use_skip

        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

        self.residual = ResidualAdd(in_ch, out_ch) if use_skip else None
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        identity = x
        out = self.act(self.norm(self.conv(x)))

        if self.use_skip:
            out = self.residual(identity, out)

        return self.pool(out)


class DepthwiseConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout, use_skip=False):
        super().__init__()
        self.use_skip = use_skip

        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

        self.residual = ResidualAdd(in_ch, out_ch) if use_skip else None
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        identity = x
        out = self.act(self.norm(self.pointwise(self.depthwise(x))))

        if self.use_skip:
            out = self.residual(identity, out)

        return self.pool(out)



class FireBlock(nn.Module):
    def __init__(self, in_ch, squeeze_ch, expand_ch, dropout, use_skip=False):
        super().__init__()
        self.use_skip = use_skip
        out_ch = 2 * expand_ch

        self.squeeze = nn.Conv2d(in_ch, squeeze_ch, 1, bias=False)
        self.expand1 = nn.Conv2d(squeeze_ch, expand_ch, 1, bias=False)
        self.expand3 = nn.Conv2d(squeeze_ch, expand_ch, 3, padding=1, bias=False)

        self.act = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm2d(out_ch)

        self.residual = ResidualAdd(in_ch, out_ch) if use_skip else None
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        identity = x
        x = self.act(self.squeeze(x))

        out = torch.cat([self.expand1(x), self.expand3(x)], dim=1)
        out = self.act(self.norm(out))

        if self.use_skip:
            out = self.residual(identity, out)

        return self.pool(out)

    

class ShuffleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups, dropout, use_skip=False):
        super().__init__()
        self.use_skip = use_skip
        mid_ch = out_ch // 4

        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, groups=groups, bias=False)
        self.dwconv = nn.Conv2d(mid_ch, mid_ch, 3, padding=1, groups=mid_ch, bias=False)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, groups=groups, bias=False)

        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.act = nn.ReLU(inplace=True)
        self.residual = ResidualAdd(in_ch, out_ch) if use_skip else None
        self.pool = nn.MaxPool2d(2)
        self.groups = groups

    def forward(self, x):
        identity = x

        out = self.act(self.bn1(self.conv1(x)))
        out = channel_shuffle(out, self.groups)
        out = self.bn2(self.dwconv(out))
        out = self.act(self.bn3(self.conv3(out)))

        if self.use_skip:
            out = self.residual(identity, out)

        return self.pool(out)





@register_model('baseline_dw')
class BaselineDepthwise(BaseModel):
    def __init__(self, num_classes: int = 8, depth: int = 5, dropout: float = 0.2, use_skip: bool = False):
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
                    dropout, 
                    use_skip=use_skip
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
        , use_skip: bool = False
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
                    dropout=dropout,
                    use_skip=use_skip
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
        , use_skip: bool = False
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
                    dropout=dropout,
                    use_skip=use_skip
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
    def __init__(self, num_classes: int = 8, depth: int = 6, dropout: float = 0.2, squeeze_ratio: float = 0.25, use_skip: bool = False):
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
                        dropout=dropout,
                        use_skip=use_skip
                    )
                )
            else:
                block_list.append(
                    DepthwiseConvBlock(
                        in_ch=input_channels,
                        out_ch=output_channels,
                        dropout=dropout,
                        use_skip=use_skip
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
    def __init__(self, num_classes: int = 8, depth: int = 6, dropout: float = 0.2, groups: int = 4, squeeze_ratio: float = 0.25, use_skip: bool = False):
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
                        dropout=dropout,
                        use_skip=use_skip
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
                        dropout=dropout,
                        use_skip=use_skip
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
    def __init__(self, num_classes: int = 8, depth: int = 6, dropout: float = 0.2, groups: int = 4, squeeze_ratio: float = 0.25, use_skip: bool = False):
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
                block_list.append(FireBlock(input_channels, squeeze_channels, expand_channels, dropout, use_skip=use_skip))
            elif i < depth - 1:
                # Shuffle middle
                g = min(groups, input_channels, output_channels)
                while input_channels % g != 0 or output_channels % g != 0:
                    g -= 1
                block_list.append(ShuffleBlock(input_channels, output_channels, groups=g, dropout=dropout, use_skip=use_skip))
            else:
                # Depthwise last
                block_list.append(DepthwiseConvBlock(input_channels, output_channels, dropout, use_skip=use_skip))
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
    def __init__(self, num_classes: int = 8, depth: int = 5, dropout: float = 0.2, use_skip: bool = False):
        super().__init__()
        block_list = []
        output_channels = 3
        for i in range(depth):
            input_channels = output_channels
            output_channels = 2 ** ((i//2) + 5)

            block_list.append(ConvBlock(input_channels, output_channels, dropout, use_skip=use_skip))
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

    

