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

        # Load pretrained InceptionV3 model
        self.backbone = models.inception_v3(
            weights="IMAGENET1K_V1",
            aux_logits=True
        )

        # Disable auxiliary classifier safely
        self.backbone.AuxLogits = None

        if pretrained:
            self.set_parameter_requires_grad(feature_extracting=pretrained)

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
