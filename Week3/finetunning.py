import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from typing import *
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from models import SimpleModel, WraperModel
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
PATIENCE = 2   # epochs without improvement
phase = 0


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


def build_optimizer(model):
    return optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
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

    torch.manual_seed(42)

    
    transformation  = F.Compose([
                                    F.ToImage(),
                                    F.ToDtype(torch.float32, scale=True),
                                    F.Resize(size=(224, 224)),
                                ])
    
    data_train = ImageFolder("/data2/users/gasbert/master/C3/2425/MIT_large_train/train", transform=transformation)
    data_val = ImageFolder("/data2/users/gasbert/master/C3/2425/MIT_large_train/test", transform=transformation) 

    #data_train = ImageFolder("/data2/users/gasbert/master/C3/2425/MIT_large_train/train")
    #data_val = ImageFolder("/data2/users/gasbert/master/C3/2425/MIT_large_train/test") 

    train_loader = DataLoader(data_train, batch_size=16, pin_memory=True, shuffle=True, num_workers=8)
    val_loader = DataLoader(data_val, batch_size=1, pin_memory=True, shuffle=False, num_workers=8)

    C, H, W = np.array(data_train[0][0]).shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = WraperModel(num_classes=8, feature_extraction=True)#SimpleModel(input_d=C*H*W, hidden_d=300, output_d=8)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model)
    num_epochs = 100

    wandb.init(
        project="inceptionv3-progressive-finetuning",
        name="inceptionv3-mit-progressive-unfreeze",
        config={
            "architecture": "InceptionV3",
            "num_classes": 8,
            "batch_size": 16,
            "optimizer": "Adam",
            "base_lr": 1e-4,
            "patience": PATIENCE,
            "unfreeze_schedule": UNFREEZE_SCHEDULE,
            "epochs": num_epochs,
        }
    )


    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    
    for epoch in tqdm.tqdm(range(num_epochs), desc="TRAINING THE MODEL"):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = test(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Test Loss: {val_loss:.4f}, Test Accuracy: {val_accuracy:.4f}")
        
        # ---- Plateau detection ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            plateau_counter = 0
        else:
            plateau_counter += 1

        # ---- Progressive unfreezing ----
        if plateau_counter >= PATIENCE and phase < len(UNFREEZE_SCHEDULE):
            print(f"\n🔓 Unfreezing blocks: {UNFREEZE_SCHEDULE[phase]}\n")

            model.unfreeze_blocks(UNFREEZE_SCHEDULE[phase])
            optimizer = build_optimizer(model)  # REBUILD optimizer

            wandb.log({
                "unfreeze/epoch": epoch,
                "unfreeze/phase": phase,
                "unfreeze/num_blocks": len(UNFREEZE_SCHEDULE[phase]),
            })

            # Optional but VERY useful
            wandb.log({
                "unfreeze/blocks": ", ".join(UNFREEZE_SCHEDULE[phase])
            })

            phase += 1
            plateau_counter = 0


        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_accuracy,
            "val/loss": val_loss,
            "val/accuracy": val_accuracy,
        })
        
    torch.save(model.state_dict(), "./saved_model.pt")

    # Plot results
    plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, {"loss": val_losses, "accuracy": val_accuracies}, "loss")
    plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, {"loss": val_losses, "accuracy": val_accuracies}, "accuracy")
