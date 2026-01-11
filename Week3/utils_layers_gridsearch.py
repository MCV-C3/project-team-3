
import torch.nn as nn
import torch
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from torchvision import models
import matplotlib.pyplot as plt

from typing import *
from torchview import draw_graph
from graphviz import Source

from PIL import Image
import torchvision.transforms.v2  as F
import numpy as np 

import pdb


class SimpleModel(nn.Module):
    def __init__(self, input_d: int, hidden_d: int, output_d: int):
        super().__init__()

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
    

class WraperModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        remove_blocks: list[str] = None,
        extra_conv_blocks: int = 0,
        classifier_depth: int = 1,
        hidden_dim: int = 512,
    ):
        super().__init__()

        remove_blocks = remove_blocks or []

        self.backbone = models.inception_v3(
            weights=None if not pretrained else "IMAGENET1K_V1",
            aux_logits=True
        )

        if pretrained:
            self.set_parameter_requires_grad(feature_extracting=pretrained)

        # Disable auxiliary classifier safely
        self.backbone.AuxLogits = None

        # ---- Disable unwanted blocks ----
        for block_name in remove_blocks:
            if hasattr(self.backbone, block_name):
                setattr(self.backbone, block_name, nn.Identity())

        # ---- Backbone output size ----
        backbone_out = 2048

        # ---- Optional extra conv blocks ----
        convs = []
        for _ in range(extra_conv_blocks):
            convs.append(nn.Conv2d(backbone_out, backbone_out, 3, padding=1))
            convs.append(nn.BatchNorm2d(backbone_out))
            convs.append(nn.ReLU(inplace=True))

        self.extra_conv = nn.Sequential(*convs) if convs else nn.Identity()

        # ---- Adaptive pooling ----
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # ---- Classifier ----
        clf = []
        in_dim = backbone_out

        for _ in range(classifier_depth - 1):
            clf.append(nn.Linear(in_dim, hidden_dim))
            clf.append(nn.ReLU(inplace=True))
            clf.append(nn.Dropout(0.5))
            in_dim = hidden_dim

        clf.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*clf)

    def forward(self, x):
        x = self.backbone.Conv2d_1a_3x3(x)
        x = self.backbone.Conv2d_2a_3x3(x)
        x = self.backbone.Conv2d_2b_3x3(x)
        x = self.backbone.maxpool1(x)

        x = self.backbone.Conv2d_3b_1x1(x)
        x = self.backbone.Conv2d_4a_3x3(x)
        x = self.backbone.maxpool2(x)

        x = self.backbone.Mixed_5b(x)
        x = self.backbone.Mixed_5c(x)
        x = self.backbone.Mixed_5d(x)

        x = self.backbone.Mixed_6a(x)
        x = self.backbone.Mixed_6b(x)
        x = self.backbone.Mixed_6c(x)
        x = self.backbone.Mixed_6d(x)
        x = self.backbone.Mixed_6e(x)

        if not isinstance(self.backbone.Mixed_7a, nn.Identity):
            x = self.backbone.Mixed_7a(x)
        if not isinstance(self.backbone.Mixed_7b, nn.Identity):
            x = self.backbone.Mixed_7b(x)
        if not isinstance(self.backbone.Mixed_7c, nn.Identity):
            x = self.backbone.Mixed_7c(x)

        x = self.extra_conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
    

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






# Example of usage
if __name__ == "__main__":
    torch.manual_seed(42)

    # Load a pretrained model and modify it
    model = WraperModel(num_classes=8, feature_extraction=False)
    #model.load_state_dict(torch.load("saved_model.pt"))
    #model = model

    """
        features.0
        features.2
        features.5
        features.7
        features.10
        features.12
        features.14
        features.17
        features.19
        features.21
        features.24
        features.26
        features.28
    """

    transformation  = F.Compose([
                                    F.ToImage(),
                                    F.ToDtype(torch.float32, scale=True),
                                    F.RandomHorizontalFlip(p=1.),
                                    F.Resize(size=(256, 256)),
                                ])
    # Example GradCAM usage
    dummy_input = Image.open("/home/cboned/data/Master/MIT_split/test/highway/art803.jpg")#torch.randn(1, 3, 224, 224)
    input_image = transformation(dummy_input).unsqueeze(0)



    target_layers = [model.backbone.features[26]]
    targets = [ClassifierOutputTarget(6)]
    
    image = torch.from_numpy(np.array(dummy_input)).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min()) ## Image needs to be between 0 and 1 and be a numpy array (Remember that if you have norlized the image you need to denormalize it before applying this (image * std + mean))

    ## VIsualize the activation map from Grad Cam
    ## To visualize this, it is mandatory to have gradients.
    
    grad_cams = model.extract_grad_cam(input_image=input_image, target_layer=target_layers, targets=targets)

    visualization = show_cam_on_image(image, grad_cams, use_rgb=True)

    # Plot the result
    plt.imshow(visualization)
    plt.axis("off")
    plt.show()

    # Display processed feature maps shapes
    feature_maps, layer_names = model.extract_feature_maps(input_image)

                                                                 ### Aggregate the feature maps
    # Process and visualize feature maps
    processed_feature_maps = []  # List to store processed feature maps
    for feature_map in feature_maps:
        feature_map = feature_map.squeeze(0)  # Remove the batch dimension
        min_feature_map, min_index = torch.min(feature_map, 0) # Get the min across channels
        processed_feature_maps.append(min_feature_map.data.cpu().numpy())
    
    
    # Plot All the convolution feature maps separately
    fig = plt.figure(figsize=(30, 50))
    for i in range(len(processed_feature_maps)):
        ax = fig.add_subplot(5, 4, i + 1)
        ax.imshow(processed_feature_maps[i], cmap="hot", interpolation="nearest")
        ax.axis("off")
        ax.set_title(f"{layer_names[i].split('(')[0]}_{i}", fontsize=10)


    plt.show()

    ## Plot a concret layer feature map when processing a image thorugh the model
    ## Is not necessary to have gradients

    with torch.no_grad():
        feature_map = (model.extract_features_from_hooks(x=input_image, layers=["features.28"]))["features.28"]
        feature_map = feature_map.squeeze(0)  # Remove the batch dimension
        print(feature_map.shape)
        processed_feature_map, _ = torch.min(feature_map, 0) 

    # Plot the result
    plt.imshow(processed_feature_map, cmap="gray")
    plt.axis("off")
    plt.show()



    ## Draw the model
    model_graph = draw_graph(model, input_size=(1, 3, 224, 224), device='meta', expand_nested=True, roll=True)
    model_graph.visual_graph.render(filename="test", format="png", directory="./Week3")