import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from typing import *
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from models import SimpleModel, PatchMlp
import torchvision.transforms.v2  as F
from torchviz import make_dot
import tqdm
import argparse


PATCH_SIZE = 32

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
    test_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Track loss and accuracy
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = test_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_by_patches(model, dataloader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
    
        batch_size, C, H, W = inputs.shape
        num_patches_h = H // PATCH_SIZE
        num_patches_w = W // PATCH_SIZE
        num_patches = num_patches_h * num_patches_w

        # Convert each image in the batch to patches
        # outputs: [batch_size, num_patches, C, PATCH_SIZE, PATCH_SIZE]
        patches = inputs.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(batch_size, num_patches, C, PATCH_SIZE, PATCH_SIZE)

        # Flatten patches to feed to MLP
        patches = patches.view(batch_size * num_patches, C*PATCH_SIZE*PATCH_SIZE)
        
        # Forward pass
        patch_outputs = model(patches)  # [batch_size * num_patches, num_classes]

        # Reshape back to [batch_size, num_patches, num_classes] for aggregation
        patch_outputs = patch_outputs.view(batch_size, num_patches, -1)
        
        # Aggregate patch predictions (e.g., mean over patches)
        outputs = patch_outputs.mean(dim=1)  # [batch_size, num_classes]

        # Compute loss on aggregated predictions
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


def train_patches(model, dataloader, criterion, optimizer, device,
                     patch_size, agg_method):
    model.train()
    train_loss, correct, total = 0.0, 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        B, C, H, W = inputs.shape
        patches = inputs.unfold(2, patch_size, patch_size)\
                        .unfold(3, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(-1, C * patch_size * patch_size)

        patch_outputs = model(patches)
        num_patches = patch_outputs.shape[0] // B
        patch_outputs = patch_outputs.view(B, num_patches, -1)

        outputs = aggregate_patches(patch_outputs, agg_method)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * B
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += B

    return train_loss / total, correct / total


def test_patches(model, dataloader, criterion, device,
                     patch_size, agg_method):
    model.eval()
    test_loss, correct, total = 0.0, 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        B, C, H, W = inputs.shape
        patches = inputs.unfold(2, patch_size, patch_size)\
                        .unfold(3, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(-1, C * patch_size * patch_size)

        patch_outputs = model(patches)
        num_patches = patch_outputs.shape[0] // B
        patch_outputs = patch_outputs.view(B, num_patches, -1)

        outputs = aggregate_patches(patch_outputs, agg_method)

        # Compute loss on aggregated predictions
        loss = criterion(outputs, labels)

        # Track loss and accuracy
        test_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return test_loss / total, correct / total



def test_by_patches(model, dataloader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            batch_size, C, H, W = inputs.shape
            num_patches_h = H // PATCH_SIZE
            num_patches_w = W // PATCH_SIZE
            num_patches = num_patches_h * num_patches_w

            # Convert each image in the batch to patches
            # outputs: [batch_size, num_patches, C, PATCH_SIZE, PATCH_SIZE]
            patches = inputs.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
            patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
            patches = patches.view(batch_size, num_patches, C, PATCH_SIZE, PATCH_SIZE)

            # Flatten patches to feed to MLP
            patches = patches.view(batch_size * num_patches, C*PATCH_SIZE*PATCH_SIZE)
            
            # Forward pass
            patch_outputs = model(patches)  # [batch_size * num_patches, num_classes]

            # Reshape back to [batch_size, num_patches, num_classes] for aggregation
            patch_outputs = patch_outputs.view(batch_size, num_patches, -1)
            
            # Aggregate patch predictions (e.g., mean over patches)
            outputs = patch_outputs.mean(dim=1)  # [batch_size, num_classes]

            # Compute loss on aggregated predictions
            loss = criterion(outputs, labels)

            # Track loss and accuracy
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = test_loss / total
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


def extract_patches(inputs, patch_size):
    batch_size, C, H, W = inputs.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w

    patches = inputs.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(batch_size * num_patches, C * patch_size * patch_size)

    return patches, num_patches


def agg_mean(patch_outputs):
    return patch_outputs.mean(dim=1)

def agg_max(patch_outputs):
    return patch_outputs.max(dim=1).values

def agg_mlp(patch_outputs, mlp):
    # patch_outputs: [B, num_patches, num_classes]
    B, P, C = patch_outputs.shape
    combined = patch_outputs.view(B, P * C)
    return mlp(combined)

def build_aggregator_head(num_patches, num_classes):
    return nn.Sequential(
        nn.Linear(num_patches * num_classes, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )

def aggregate_patches(patch_outputs, method="mean"):
    if method == "mean":
        return patch_outputs.mean(dim=1)
    elif method == "max":
        return patch_outputs.max(dim=1).values
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


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

    parser = argparse.ArgumentParser(description="Train a SimpleModel on image dataset.")
    parser.add_argument("--patches", action="store_true", help="Use patch-based training/testing")
    args = parser.parse_args()

    torch.manual_seed(42)

    transformation  = F.Compose([
                                    F.ToImage(),
                                    F.ToDtype(torch.float32, scale=True),
                                    F.Resize(size=(224, 224)),
                                ])
    
    data_train = ImageFolder("/data2/users/gasbert/master/C3/places_reduced/train", transform=transformation)
    data_test = ImageFolder("/data2/users/gasbert/master/C3/places_reduced/val", transform=transformation) 

    all_labels = [label for _, label in data_train]
    print("Train labels min/max:", min(all_labels), max(all_labels))

    train_loader = DataLoader(data_train, batch_size=256, pin_memory=True, shuffle=True, num_workers=8)
    test_loader = DataLoader(data_test, batch_size=128, pin_memory=True, shuffle=False, num_workers=8)

    C, H, W = np.array(data_train[0][0]).shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device used: ", device)
    
    if args.patches:
        print("Using Patches Model")
        model = PatchMlp(input_d=C*PATCH_SIZE*PATCH_SIZE, hidden_d=300, output_d=11)
    else:
        print("Using Baseline Model")
        model = SimpleModel(input_d=C*H*W, hidden_d=300, output_d=11)

    if args.patches:
        plot_computational_graph(model, input_size=(1, C*PATCH_SIZE*PATCH_SIZE))
    else:
        plot_computational_graph(model, input_size=(1, C*H*W))

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 30

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    
    for epoch in tqdm.tqdm(range(num_epochs), desc="TRAINING THE MODEL"):
        
        if args.patches:
            train_loss, train_accuracy = train_by_patches(model, train_loader, criterion, optimizer, device)
            test_loss, test_accuracy = test_by_patches(model, test_loader, criterion, device)
        else:
            train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
            test_loss, test_accuracy = test(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Plot results
    plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, {"loss": test_losses, "accuracy": test_accuracies}, "loss")
    plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, {"loss": test_losses, "accuracy": test_accuracies}, "accuracy")

    print("Highest Test Accuracy: ", max(test_accuracies))
