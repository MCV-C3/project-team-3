from logging import config
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from typing import *
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from utils_gridsearch import SimpleModel, WraperModel
import torchvision.transforms.v2  as F
from torchviz import make_dot
import tqdm
import wandb
from torchvision.transforms import ColorJitter
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


UNFREEZE_SCHEDULE = [
    ["Mixed_7c"],                      # phase 1
    ["Mixed_7b", "Mixed_7c"],           # phase 2
    ["Mixed_7a", "Mixed_7b", "Mixed_7c"],
]

CROSS_VAL = False
CV_FOLDS = [
    "/data2/users/gasbert/master/C3/2425/MIT_small_train_1",
    "/data2/users/gasbert/master/C3/2425/MIT_small_train_2",
    "/data2/users/gasbert/master/C3/2425/MIT_small_train_3",
    "/data2/users/gasbert/master/C3/2425/MIT_small_train_4",
]


NUM_EPOCHS = 60


# Train function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss and accuracy
        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = train_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def test(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Track loss and accuracy
            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = val_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def plot_metrics(train_metrics: Dict, test_metrics: Dict, metric_name: str):
    """
    Plots and saves metrics for training and testing.

    Args:
        train_metrics (Dict): Dictionary containing training metrics.
        test_metrics (Dict): Dictionary containing testing metrics.
        metric_name (str): The name of the metric to plot (e.g., "loss", "accuracy").

    Saves:
        - loss.png for loss plots
        - metrics.png for other metrics plots
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics[metric_name], label=f'Train {metric_name.capitalize()}')
    plt.plot(test_metrics[metric_name], label=f'Test {metric_name.capitalize()}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.capitalize())
    plt.title(f'{metric_name.capitalize()} Over Epochs')
    plt.legend()
    plt.grid(True)

    # Save the plot with the appropriate name
    filename = "loss.png" if metric_name.lower() == "loss" else "metrics.png"
    plt.savefig(filename)
    print(f"Plot saved as {filename}")

    plt.close()  # Close the figure to free memory



def get_dataloaders(train_root, test_root, batch_size=16, input_size=224):
    
    # Best data augmentation configuration
    train_transformation = F.Compose([
        F.ToImage(),
        F.Resize((input_size, input_size)),
        F.RandomHorizontalFlip(p=0.5),
        F.RandomRotation(degrees=10),
        ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
        F.ToDtype(torch.float32, scale=True),
        F.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_transformation = F.Compose([
        F.ToImage(),
        F.Resize((input_size, input_size)),
        F.ToDtype(torch.float32, scale=True),
        F.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    train_dataset = ImageFolder(os.path.join(train_root, "train"), transform=train_transformation)
    test_dataset  = ImageFolder(os.path.join(test_root, "test"), transform=val_transformation)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    return train_loader, test_loader


def build_optimizer(model, lr):
    return optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )


def save_gradcam_for_loader(
    model,
    dataloader,
    device,
    epoch,
    save_root="gradcam_outputs",
    target_layer_name="Mixed_7c",
    hook_layer_name="Mixed_7c",
):
    """
    Generates and saves:
    1) Grad-CAM visualization
    2) Hook-based feature map (min over channels)
    for ALL images in a dataloader.
    """

    model.eval()
    os.makedirs(save_root, exist_ok=True)

    # ---- Grad-CAM target layer ----
    target_layer = dict(model.backbone.named_children())[target_layer_name]

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        for i in range(inputs.size(0)):
            input_tensor = inputs[i].unsqueeze(0)
            label = labels[i].item()

            # ======================================================
            # 1️⃣ GRAD-CAM
            # ======================================================
            input_tensor.requires_grad_(True)

            targets = [ClassifierOutputTarget(label)]

            grad_cam = model.extract_grad_cam(
                input_image=input_tensor,
                target_layer=[target_layer],
                targets=targets
            )

            # De-normalize image for visualization
            img = inputs[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            cam_vis = show_cam_on_image(img, grad_cam, use_rgb=True)

            # Get original image name
            dataset = dataloader.dataset
            img_path, class_idx = dataset.samples[batch_idx * dataloader.batch_size + i]
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            class_name = dataset.classes[class_idx]

            cam_path = os.path.join(
                save_root,
                f"class_{img_name}_image_{img_name}_epoch_{epoch}.png"
            )
            plt.imsave(cam_path, cam_vis)



def run_training(train_loader, val_loader, config, fold_id=None):
    best_val_loss = float("inf")
    plateau_counter = 0
    phase = 0
    patience = config.patience
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WraperModel(
        num_classes=8,
        pretrained=True,
        head_depth=config.head_depth,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        use_batchnorm=config.use_batchnorm,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config.lr)

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in tqdm.tqdm(range(NUM_EPOCHS), desc=f"Fold {fold_id} Training"):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = test(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, "
              f"Test Loss: {val_loss:.4f}, Test Accuracy: {val_acc:.4f}")
        

        if (epoch + 1) % 10 == 1:
            print(f"\n🎯 Saving Grad-CAMs at epoch {epoch + 1}\n")
            save_gradcam_for_loader(
                model=model,
                dataloader=train_loader,
                device=device,
                epoch=epoch + 1,
                save_root="gradcam_outputs",
                target_layer_name="Mixed_7c",
            )
        
        # ---- Plateau detection ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            plateau_counter = 0
        else:
            plateau_counter += 1

        # ---- Progressive unfreezing ----
        if plateau_counter >= patience and phase < len(UNFREEZE_SCHEDULE):
            print(f"\n🔓 Unfreezing blocks: {UNFREEZE_SCHEDULE[phase]}\n")

            model.unfreeze_blocks(UNFREEZE_SCHEDULE[phase])
            optimizer = build_optimizer(model, config.lr)  # REBUILD optimizer

            wandb.log({
                "epoch": epoch,
                "unfreeze/epoch_loss": 2,
                "unfreeze/epoch_accuracy": 1,
                "unfreeze/phase": phase,
                "unfreeze/num_blocks": len(UNFREEZE_SCHEDULE[phase]),
            })

            # Optional but VERY useful
            wandb.log({
                "unfreeze/blocks": ", ".join(UNFREEZE_SCHEDULE[phase])
            })

            phase += 1
            plateau_counter = 0
        else:
            if phase < len(UNFREEZE_SCHEDULE):
                wandb.log({
                    "epoch": epoch,
                    "unfreeze/epoch_loss": 0,
                    "unfreeze/epoch_accuracy": 0,
                    "unfreeze/phase": phase,
                    "unfreeze/num_blocks": len(UNFREEZE_SCHEDULE[phase]),
                })
            else:
                wandb.log({
                    "epoch": epoch,
                    "unfreeze/epoch_loss": 0,
                    "unfreeze/epoch_accuracy": 0,
                    "unfreeze/phase": phase,
                    "unfreeze/num_blocks": len(UNFREEZE_SCHEDULE[phase-1]),
                })
        

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
        })

    return {
        "train_loss": train_losses,
        "train_acc": train_accuracies,
        "val_loss": val_losses,
        "val_acc": val_accuracies,
        "final_val_acc": val_accuracies[-1],
        "final_val_loss": val_losses[-1]
    }


def sweep_train():
    wandb.init()
    config = wandb.config

    torch.manual_seed(42)

    # ---- SINGLE FOLD PER SWEEP RUN ----
    fold_path = CV_FOLDS[0]   # IMPORTANT: do NOT cross-val inside a sweep

    train_loader, val_loader = get_dataloaders(
        train_root=fold_path,
        test_root=fold_path,
        batch_size=config.batch_size,
        input_size=config.input_size
    )

    run_training(
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        fold_id=0,
    )


def plot_computational_graph(model: torch.nn.Module, input_size: tuple, filename: str = "computational_graph"):
    """
    Generates and saves a plot of the computational graph of the model.

    Args:
        model (torch.nn.Module): The PyTorch model to visualize.
        input_size (tuple): The size of the dummy input tensor (e.g., (batch_size, input_dim)).
        filename (str): Name of the file to save the graph image.
    """
    model.eval()  # Set the model to evaluation mode
    
    # Generate a dummy input based on the specified input size
    dummy_input = torch.randn(*input_size)

    # Create a graph from the model
    graph = make_dot(model(dummy_input), params=dict(model.named_parameters()), show_attrs=True).render(filename, format="png")

    print(f"Computational graph saved as {filename}")




if __name__ == "__main__":

    sweep_train()