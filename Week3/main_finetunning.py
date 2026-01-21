import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from typing import *
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from utils import SimpleModel, WraperModel
import torchvision.transforms.v2  as F
from torchviz import make_dot
import tqdm
import wandb


from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomResizedCrop, ColorJitter


UNFREEZE_SCHEDULE = [
    ["Mixed_7c"],                      # phase 1
    ["Mixed_7b", "Mixed_7c"],           # phase 2
    ["Mixed_7a", "Mixed_7b", "Mixed_7c"],
]


PATIENCE = 4   # epochs without improvement

CROSS_VAL = True
CV_FOLDS = [
    "/data2/users/gasbert/master/C3/2425/MIT_small_train_1",
    #"/data2/users/gasbert/master/C3/2425/MIT_small_train_2",
    #"/data2/users/gasbert/master/C3/2425/MIT_small_train_3",
    #"/data2/users/gasbert/master/C3/2425/MIT_small_train_4",
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


def get_dataloaders(train_root, test_root, batch_size=16):
    # Best data augmentation configuration
    train_transformation = F.Compose([
        F.ToImage(),
        F.Resize((224, 224)),
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
        F.Resize((224, 224)),
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


def run_training(train_loader, val_loader, fold_id=None):

    best_val_acc = 0.0
    best_epoch = -1
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("inf")
    plateau_counter = 0
    phase = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WraperModel(
        num_classes=8,
        pretrained=True
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model)

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in tqdm.tqdm(range(num_epochs), desc=f"Fold {fold_id} Training"):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = test(model, val_loader, criterion, device)

        # ---- Save best model by validation accuracy ----
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_epoch = epoch

            checkpoint = {
                "epoch": epoch,
                "val_accuracy": val_accuracy,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_classes": 8,
                "architecture": "InceptionV3",
                "classifier_head": model.backbone.fc,
            }

            torch.save(
                checkpoint,
                os.path.join(save_dir, f"best_model_fold_{fold_id}.pth")
            )

            print(f"💾 Saved best model (epoch {epoch}, val acc {val_accuracy:.4f})")

            wandb.log({
                "best/val_accuracy": val_accuracy,
                "best/epoch": epoch
            })

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
            "train/accuracy": train_accuracy,
            "val/loss": val_loss,
            "val/accuracy": val_accuracy,
        })

    return {
        "train_loss": train_losses,
        "train_acc": train_accuracies,
        "val_loss": val_losses,
        "val_acc": val_accuracies,
        "final_val_acc": val_accuracies[-1],
        "final_val_loss": val_losses[-1]
    }

 

if __name__ == "__main__":

    torch.manual_seed(42)
    num_epochs = 100

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
                project="inceptionv3-finetuning-dropout",
                name=f"inceptionv3-fold-{fold_idx}",
                config={
                    "architecture": "InceptionV3",
                    "num_classes": 8,
                    "batch_size": 16,
                    "optimizer": "Adam",
                    "lr": 1e-4,
                    "patience": PATIENCE,
                    "unfreeze_schedule": UNFREEZE_SCHEDULE,
                    "epochs": num_epochs,
                    "fold": fold_idx,
                    "pretrained": True,
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
            name="inceptionv3-mit-progressive-unfreeze_better",
            config={
                    "architecture": "InceptionV3",
                    "num_classes": 8,
                    "batch_size": 16,
                    "optimizer": "Adam",
                    "lr": 1e-4,
                    "patience": PATIENCE,
                    "unfreeze_schedule": UNFREEZE_SCHEDULE,
                    "epochs": num_epochs,
                    "pretrained": True,
                },
        )

        run_training(train_loader, val_loader)