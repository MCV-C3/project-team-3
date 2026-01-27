"""
Final Best Model Architecture from Ablation Study
Model: SE_R4_6Layer
Validation Accuracy: 91.52%
Test Accuracy: 90.83%
Parameters: 2,078,344
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


from torchvision import models
from typing import *
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Model(nn.Module):
    def __init__(self, num_classes=8):
        super(Model, self).__init__()
        
        self.channels = [32, 64, 128, 128, 256, 512]
        self.num_conv_layers = 6
        
        # Build convolutional layers with residual connections
        self.conv_blocks = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.se_blocks = nn.ModuleList()
        self.residual_projs = nn.ModuleList()
        
        in_channels = 3
        for i in range(self.num_conv_layers):
            out_channels = self.channels[i]
            
            # Convolutional block
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                nn.Dropout2d(p=0.2)
            )
            self.conv_blocks.append(conv_block)
            
            # SE block
            self.se_blocks.append(SEBlock(out_channels, reduction=4))
            
            # Pooling (skip some layers for deep networks)
            if i not in [4, 5]:  # Pool at layers 0,1,2,3 only
                self.pool_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                self.pool_layers.append(nn.Identity())
            
            # Residual projection if channels change
            if i > 0 and in_channels != out_channels:
                self.residual_projs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
            else:
                self.residual_projs.append(None)
            
            in_channels = out_channels
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Convolutional layers with residual connections and SE blocks
        for i in range(self.num_conv_layers):
            identity = x
            
            # Conv block
            x = self.conv_blocks[i](x)
            
            # SE block
            x = self.se_blocks[i](x)
            
            # Residual connection (skip first layer)
            if i > 0:
                # Project identity if needed
                if self.residual_projs[i] is not None:
                    identity = self.residual_projs[i](identity)
                
                # Match spatial dimensions if needed
                if identity.shape[2:] != x.shape[2:]:
                    identity = F.adaptive_avg_pool2d(identity, x.shape[2:])
                
                x = x + identity
            
            # Pooling
            x = self.pool_layers[i](x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
    
    def get_num_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



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



def create_final_model(num_classes=8):
    """Create the final optimized model."""
    return Model(num_classes=num_classes)


def create_model_configurable(model_name='configurable_cnn', num_classes=8, **kwargs):
    """
    Create a model based on configuration.
    
    Args:
        model_name (str): Name of the model
        num_classes (int): Number of output classes
        **kwargs: Model-specific parameters
        
    Returns:
        torch.nn.Module: Model instance
    """
    if model_name == 'configurable_cnn' or model_name == 'advanced_cnn':
        return ConfigurableCNN(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


class ConfigurableCNN(nn.Module):
    """
    Fully configurable CNN using only official PyTorch layers.
    Minimum 2 convolutional layers required.
    """
    def __init__(self, num_classes=8, **config):
        super(ConfigurableCNN, self).__init__()
        
        # Extract configuration
        self.num_conv_layers = config.get('num_conv_layers', 2)
        self.channels = config.get('channels', [32, 64])
        self.norm_type = config.get('norm_type', 'BatchNorm2d')
        self.num_groups = config.get('num_groups', 2)
        self.activation = config.get('activation', 'ReLU')
        self.pool_type = config.get('pool_type', 'MaxPool2d')
        self.dropout_conv = config.get('dropout_conv', 0.0)
        self.dropout_fc = config.get('dropout_fc', 0.0)
        self.use_attention = config.get('use_attention', False)
        self.num_heads = config.get('num_heads', 4)
        self.recurrent_type = config.get('recurrent_type', None)
        self.hidden_size = config.get('hidden_size', 128)
        self.use_residual = config.get('use_residual', False)
        self.residual_type = config.get('residual_type', 'simple')
        self.use_se = config.get('use_se', False)
        self.se_reduction = config.get('se_reduction', 16)
        
        assert self.num_conv_layers >= 2, "Minimum 2 convolutional layers required"
        assert len(self.channels) >= self.num_conv_layers, "Not enough channel specifications"
        
        # Build convolutional layers
        self.conv_blocks = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.se_blocks = nn.ModuleList() if self.use_se else None
        
        in_channels = 3
        for i in range(self.num_conv_layers):
            out_channels = self.channels[i]
            
            # Conv block
            layers = []
            
            # Main convolution
            if self.use_residual and self.residual_type == 'bottleneck' and i > 0:
                # Bottleneck: 1x1 -> 3x3 -> 1x1
                mid_channels = out_channels // 4
                layers.extend([
                    nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                    self._get_norm_layer(mid_channels),
                    self._get_activation(),
                    nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                    self._get_norm_layer(mid_channels),
                    self._get_activation(),
                    nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
                ])
            else:
                # Standard convolution
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
            
            # Normalization
            if self.norm_type:
                layers.append(self._get_norm_layer(out_channels))
            
            # Activation
            layers.append(self._get_activation())
            
            # Dropout after conv
            if self.dropout_conv > 0:
                layers.append(nn.Dropout2d(p=self.dropout_conv))
            
            self.conv_blocks.append(nn.Sequential(*layers))
            
            # Pooling - reduce frequency for deep networks to avoid dimension collapse
            # For 128x128 input: can pool max 7 times (128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1)
            # Strategy: pool every layer for first 4 layers, then every 2 layers
            if self.pool_type == 'mixed':
                pool = nn.MaxPool2d(2) if i % 2 == 0 else nn.AvgPool2d(2)
            elif self.pool_type == 'AdaptiveAvgPool2d':
                pool = nn.Identity()  # No pooling between layers, only at end
            else:
                pool = self._get_pool_layer()
            
            # Skip pooling for deep networks to prevent collapse
            # Pool only on certain layers based on depth
            if self.num_conv_layers <= 5:
                # Shallow networks: pool after every layer
                self.pool_layers.append(pool)
            elif self.num_conv_layers <= 7:
                # Medium networks: pool after layers 0,1,2,3,5 (skip 4,6)
                if i not in [4, 6]:
                    self.pool_layers.append(pool)
                else:
                    self.pool_layers.append(nn.Identity())
            else:
                # Deep networks (8+): pool after layers 0,1,2,4,6 (skip 3,5,7+)
                if i in [0, 1, 2, 4, 6]:
                    self.pool_layers.append(pool)
                else:
                    self.pool_layers.append(nn.Identity())
            
            # SE block
            if self.use_se:
                self.se_blocks.append(SEBlock(out_channels, self.se_reduction))
            
            # Residual projection if needed
            if self.use_residual and in_channels != out_channels:
                setattr(self, f'residual_proj_{i}', 
                       nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
            
            in_channels = out_channels
        
        # Global pooling
        if self.pool_type == 'AdaptiveAvgPool2d' or self.pool_type == 'adaptive':
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Attention layer
        if self.use_attention:
            # Ensure num_heads divides embed_dim
            embed_dim = self.channels[self.num_conv_layers-1]
            # Adjust num_heads if it doesn't divide embed_dim
            if embed_dim % self.num_heads != 0:
                # Find closest divisor
                for h in range(self.num_heads, 0, -1):
                    if embed_dim % h == 0:
                        self.num_heads = h
                        break
            self.attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=self.num_heads,
                batch_first=True
            )
        
        # Recurrent layer
        if self.recurrent_type == 'LSTM':
            self.recurrent = nn.LSTM(
                input_size=self.channels[self.num_conv_layers-1],
                hidden_size=self.hidden_size,
                batch_first=True
            )
            fc_input_size = self.hidden_size
        elif self.recurrent_type == 'GRU':
            self.recurrent = nn.GRU(
                input_size=self.channels[self.num_conv_layers-1],
                hidden_size=self.hidden_size,
                batch_first=True
            )
            fc_input_size = self.hidden_size
        else:
            fc_input_size = self.channels[self.num_conv_layers-1]
        
        # Classifier
        classifier_layers = []
        if self.dropout_fc > 0:
            classifier_layers.append(nn.Dropout(p=self.dropout_fc))
        classifier_layers.append(nn.Linear(fc_input_size, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
    
    def _get_norm_layer(self, channels):
        if self.norm_type == 'BatchNorm2d':
            return nn.BatchNorm2d(channels)
        elif self.norm_type == 'LayerNorm':
            # LayerNorm for 2D: normalize over C, H, W dimensions
            # We'll use GroupNorm with num_groups=1 as a workaround for LayerNorm on conv features
            return nn.GroupNorm(1, channels)
        elif self.norm_type == 'GroupNorm':
            return nn.GroupNorm(self.num_groups, channels)
        else:
            return nn.Identity()
    
    def _get_activation(self):
        if self.activation == 'ReLU':
            return nn.ReLU(inplace=True)
        elif self.activation == 'LeakyReLU':
            return nn.LeakyReLU(inplace=True)
        elif self.activation == 'ELU':
            return nn.ELU(inplace=True)
        elif self.activation == 'GELU':
            return nn.GELU()
        else:
            return nn.ReLU(inplace=True)
    
    def _get_pool_layer(self):
        if self.pool_type == 'MaxPool2d':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif self.pool_type == 'AvgPool2d':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            return nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        # Convolutional layers
        for i in range(self.num_conv_layers):
            identity = x
            
            # Apply conv block
            x = self.conv_blocks[i](x)
            
            # Apply SE block
            if self.use_se:
                x = self.se_blocks[i](x)
            
            # Apply residual connection
            if self.use_residual and i > 0:
                if hasattr(self, f'residual_proj_{i}'):
                    identity = getattr(self, f'residual_proj_{i}')(identity)
                # Match spatial dimensions if needed
                if identity.shape[2:] != x.shape[2:]:
                    identity = F.adaptive_avg_pool2d(identity, x.shape[2:])
                x = x + identity
            
            # Apply pooling
            x = self.pool_layers[i](x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Attention
        if self.use_attention:
            try:
                x = x.unsqueeze(1)  # Add sequence dimension
                x, _ = self.attention(x, x, x)
                x = x.squeeze(1)
            except RuntimeError:
                # Skip attention if there's a dimension mismatch
                pass
        
        # Recurrent
        if self.recurrent_type:
            try:
                x = x.unsqueeze(1)  # Add sequence dimension
                x, _ = self.recurrent(x)
                x = x.squeeze(1)
            except RuntimeError:
                # Skip recurrent if there's an error
                pass
        
        # Classifier
        x = self.classifier(x)
        
        return x
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



if __name__ == "__main__":
    # Test model
    model = create_final_model(num_classes=8)
    print(f"Model created successfully!")
    print(f"Total parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    x = torch.randn(4, 3, 128, 128)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Expected output shape: (4, 8)")
