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
import argparse

from models import SimpleModel
from main import train_patches, test_patches, agg_mean, agg_max, agg_mlp, build_aggregator_head


patch_sizes = [16, 32, 56]
aggregation_methods = {
    "mean": agg_mean,
    "max": agg_max
}


if __name__ == "__main__":

    torch.manual_seed(42)

    transformation = F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.Resize(size=(224, 224)),
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
        data_train, batch_size=256,
        pin_memory=True, shuffle=True, num_workers=8
    )
    test_loader = DataLoader(
        data_test, batch_size=128,
        pin_memory=True, shuffle=False, num_workers=8
    )

    C, H, W = np.array(data_train[0][0]).shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device used:", device)

    criterion = nn.CrossEntropyLoss()

    num_epochs = 10   # keep small for grid search

    results = []

    for patch_size in patch_sizes:
        for agg_method in aggregation_methods:

            print("\n===================================")
            print(f"PATCH SIZE: {patch_size} | AGG: {agg_method}")
            print("===================================")

            model = SimpleModel(
                input_d=C * patch_size * patch_size,
                hidden_d=300,
                output_d=11
            ).to(device)

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

                print(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Train Acc: {train_acc:.4f} | "
                    f"Test Acc: {test_acc:.4f}"
                )

            best_acc = max(test_accs)
            results.append((patch_size, agg_method, best_acc))

            print(f"BEST TEST ACC: {best_acc:.4f}")

    print("\n======= GRID SEARCH RESULTS =======")
    results.sort(key=lambda x: x[2], reverse=True)

    for ps, agg, acc in results:
        print(f"Patch {ps:>3}px | Agg {agg:<4} | Best Acc {acc:.4f}")