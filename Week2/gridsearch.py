import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.v2  as F
from torchviz import make_dot
import tqdm
import wandb

from models import SimpleModel, PatchMlp
from main import agg_mean, agg_max, agg_mlp, build_aggregator_head


patch_sizes = [4, 8, 16, 32, 64]
aggregation_methods = {
    "mean": agg_mean,
    "max": agg_max
}


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


def aggregate_patches(patch_outputs, method="mean"):
    if method == "mean":
        return patch_outputs.mean(dim=1)
    elif method == "max":
        return patch_outputs.max(dim=1).values
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


if __name__ == "__main__":

    torch.manual_seed(42)

    transformation = F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.Resize(size=(128, 128)),
    ])

    data_train = ImageFolder(
        "/data2/users/gasbert/master/C3/places_reduced/train",
        transform=transformation
    )
    data_test = ImageFolder(
        "/data2/users/gasbert/master/C3/places_reduced/val",
        transform=transformation
    )

    train_loader = DataLoader(
        data_train, batch_size=1028,
        pin_memory=True, shuffle=True, num_workers=8
    )
    test_loader = DataLoader(
        data_test, batch_size=1028,
        pin_memory=True, shuffle=False, num_workers=8
    )

    C, H, W = np.array(data_train[0][0]).shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device used:", device)

    criterion = nn.CrossEntropyLoss()

    num_epochs = 60 

    results = []

    for patch_size in patch_sizes:
        for agg_method in aggregation_methods:

            run = wandb.init(
                project="patch-grid-search_final",
                name=f"patch{patch_size}_agg{agg_method}",
                config={
                    "patch_size": patch_size,
                    "aggregation": agg_method,
                    "hidden_dim": 300,
                    "learning_rate": 1e-3,
                    "epochs": num_epochs,
                    "batch_size_train": 1028,
                    "batch_size_test": 1028,
                    "model": "SimpleModel"
                },
                reinit=True
            )

            print("\n===================================")
            print(f"PATCH SIZE: {patch_size} | AGG: {agg_method}")
            print("===================================")

            # Using Shinto's best result (2 layers, 128 neurons each)
            model = PatchMlp(
                input_d=C * patch_size * patch_size,
                hidden_dims=[128, 128],
                output_d=11
            ).to(device)
            wandb.watch(model, log="gradients", log_freq=100)

            optimizer = optim.Adam(model.parameters(), lr=1e-3)

            train_accs, test_accs = [], []

            for epoch in tqdm.tqdm(range(num_epochs), desc="Training"):
                train_loss, train_acc = train_patches(
                    model, train_loader, criterion, optimizer,
                    device, patch_size, agg_method
                )

                test_loss, test_acc = test_patches(
                    model, test_loader, criterion,
                    device, patch_size, agg_method
                )

                train_accs.append(train_acc)
                test_accs.append(test_acc)

                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "test_loss": test_loss,
                    "test_accuracy": test_acc
                })

                print(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Train Acc: {train_acc:.4f} | "
                    f"Test Acc: {test_acc:.4f}"
                )

            best_acc = max(test_accs)
            wandb.summary["best_test_accuracy"] = best_acc
            results.append((patch_size, agg_method, best_acc))

            print(f"BEST TEST ACC: {best_acc:.4f}")
            wandb.finish()

    print("\n======= GRID SEARCH RESULTS =======")
    results.sort(key=lambda x: x[2], reverse=True)


    table = wandb.Table(columns=["patch_size", "aggregation", "best_test_accuracy"])

    for ps, agg, acc in results:
        table.add_data(ps, agg, acc)
        print(f"Patch {ps:>3}px | Agg {agg:<4} | Best Acc {acc:.4f}")

    wandb.init(project="patch-grid-search", name="summary", reinit=True)
    wandb.log({"grid_search_results": table})
    wandb.finish()
        