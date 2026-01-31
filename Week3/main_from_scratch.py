import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from typing import *
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from Week3.utils import SimpleModel, WraperModel
import torchvision.transforms.v2  as F
from torchviz import make_dot
import tqdm
import wandb


from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomResizedCrop


UNFREEZE_SCHEDULE = [
    ["Mixed_7c"],                      # phase 1
    ["Mixed_7b", "Mixed_7c"],           # phase 2
    ["Mixed_7a", "Mixed_7b", "Mixed_7c"],
]

best_val_loss = float("inf")
plateau_counter = 0
PATIENCE = 0   # epochs without improvement
phase = 0

CROSS_VAL = True
CV_FOLDS = [
    "/data2/users/gasbert/master/C3/2425/MIT_small_train_1",
    "/data2/users/gasbert/master/C3/2425/MIT_small_train_2",
    "/data2/users/gasbert/master/C3/2425/MIT_small_train_3",
    "/data2/users/gasbert/master/C3/2425/MIT_small_train_4",
]


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

# Data augmentation example
def get_data_transforms():
    """
    Returns a Compose object with data augmentation transformations.
    """
    return Compose([
        RandomResizedCrop(size=224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_dataloaders(train_root, test_root, batch_size=16):
    transformation = F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.Resize(size=(224, 224)),
    ])

    train_dataset = ImageFolder(os.path.join(train_root, "train"), transform=transformation)
    test_dataset  = ImageFolder(os.path.join(test_root, "test"), transform=transformation)

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


def run_training(train_loader, val_loader, fold_id=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WraperModel(
        num_classes=8,
        pretrained=False
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in tqdm.tqdm(range(num_epochs), desc=f"Fold {fold_id} Training"):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = test(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, "
              f"Test Loss: {val_loss:.4f}, Test Accuracy: {val_acc:.4f}")
        

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

    torch.manual_seed(42)
    num_epochs = 60

    if CROSS_VAL:
        all_fold_results = []

        for fold_idx, fold_path in enumerate(CV_FOLDS, start=1):
            print(f"\n===== STARTING FOLD {fold_idx} =====")

            train_loader, val_loader = get_dataloaders(
                train_root=fold_path,
                test_root=fold_path,
                batch_size=16
            )

            wandb.init(
                project="inceptionv3-crossval",
                name=f"inceptionv3-fold-{fold_idx}",
                config={
                    "architecture": "InceptionV3",
                    "num_classes": 8,
                    "batch_size": 16,
                    "optimizer": "Adam",
                    "lr": 3e-4,
                    "epochs": num_epochs,
                    "fold": fold_idx,
                    "pretrained": False,
                },
                reinit=True
            )

            fold_results = run_training(
                train_loader,
                val_loader,
                fold_id=fold_idx
            )

            all_fold_results.append(fold_results)
            wandb.finish()

        # ---- Aggregate results ----
        mean_acc = np.mean([r["final_val_acc"] for r in all_fold_results])
        std_acc  = np.std([r["final_val_acc"] for r in all_fold_results])

        print("\n===== CROSS-VALIDATION RESULTS =====")
        print(f"Mean Validation Accuracy: {mean_acc:.4f}")
        print(f"Std Validation Accuracy : {std_acc:.4f}")

    else:
        # ---- Original single-split training ----
        train_loader, val_loader = get_dataloaders(
            "/data2/users/gasbert/master/C3/2425/MIT_large_train",
            "/data2/users/gasbert/master/C3/2425/MIT_large_train",
            batch_size=16
        )

        wandb.init(
            project="inceptionv3-progressive-finetuning",
            name="inceptionv3-mit-from-scratch",
            config={
                    "architecture": "InceptionV3",
                    "num_classes": 8,
                    "batch_size": 16,
                    "optimizer": "Adam",
                    "lr": 3e-4,
                    "epochs": num_epochs,
                    "pretrained": False,
                },
        )

        run_training(train_loader, val_loader)