import argparse
import os

import pandas as pd
import torch
import wandb
import yaml
from dataset import C3Dataset
from Week3.utils import WraperModel
from torch.nn import CrossEntropyLoss
from torch.optim import Adagrad, Adam, AdamW, Optimizer, RMSprop
from torch.utils.data import DataLoader
from torchvision.models.inception import InceptionOutputs
from torchvision.transforms import (Compose, RandomHorizontalFlip,
                                    RandomRotation, Resize)
from tqdm import tqdm


def parse_config(cfg_file: str) -> argparse.Namespace:
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def make_dataset(path: str, device: torch.device | str,
        batch_size: int = 8,
        num_workers: int = 0
    ) -> tuple[DataLoader, DataLoader]:
    train_transform = Compose([
        RandomRotation(20),
        RandomHorizontalFlip(),
        Resize([299, 299])
    ])
    train_dataset = C3Dataset(path, train=True, transform=train_transform, device=device)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    test_transform = Compose([
        Resize([299, 299])
    ])
    test_dataset = C3Dataset(path, train=False, transform=test_transform, device=device)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_dataloader, test_dataloader


def make_optimizer(optimizer: str, model: WraperModel, lr: int) -> Optimizer:
    opt = None
    match optimizer:
        case 'adagrad': opt = Adagrad(model.parameters(), lr=lr)
        case 'rmsprop': opt = RMSprop(model.parameters(), lr=lr)
        case 'adam': opt = Adam(model.parameters(), lr=lr)
        case 'adamw': opt = AdamW(model.parameters(), lr=lr)
        case _: raise ValueError(f"Optimizer {optimizer} not implemented. Either add it or use another one.")
    return opt


def train(model: WraperModel, dataloader: DataLoader, criterion: CrossEntropyLoss, optimizer: Optimizer):
    model.train()
    mean_loss = 0.0
    mean_accuracy = 0.0

    for images, labels in dataloader:
        optimizer.zero_grad()

        output: torch.Tensor = model(images)
        loss: torch.Tensor = criterion(output, labels)
        loss.backward()
        optimizer.step()

        mean_loss += loss * labels.shape[0]
        _, predicted = output.max(1)
        mean_accuracy += torch.sum(predicted == labels)
    
    mean_loss /= len(dataloader.dataset)
    mean_accuracy /= len(dataloader.dataset)

    # Image log
    img_list = []
    for img, output, label in zip(images[-4:], predicted[-4:], labels[-4:]):
        img = img.cpu().detach()
        img = img.permute(1, 2, 0).numpy()
        caption = f"Output: {dataloader.dataset.classes[output.item()]} \nLabel: {dataloader.dataset.classes[label.item()]}"
        img_list.append(wandb.Image(img, caption=caption))

    return mean_loss.item(), mean_accuracy.item(), img_list


@torch.no_grad()
def test(model: WraperModel, dataloader: DataLoader, criterion: CrossEntropyLoss):
    model.eval()
    mean_loss = 0.0
    mean_accuracy = 0.0

    for images, labels in dataloader:
        output: torch.Tensor = model(images)
        loss: torch.Tensor = criterion(output, labels)

        mean_loss += loss * labels.shape[0]
        _, predicted = output.max(1)
        mean_accuracy += torch.sum(predicted == labels)
    
    mean_loss /= len(dataloader.dataset)
    mean_accuracy /= len(dataloader.dataset)

    # Image log
    img_list = []
    for img, output, label in zip(images[-4:], predicted[-4:], labels[-4:]):
        img = img.cpu().detach()
        img = img.permute(1, 2, 0).numpy()
        caption = f"Output: {dataloader.dataset.classes[output.item()]}\nLabel: {dataloader.dataset.classes[label.item()]}"
        img_list.append(wandb.Image(img, caption=caption))

    return mean_loss.item(), mean_accuracy.item(), img_list


def main_training(cfg: dict, cfg_naming: dict) -> tuple[float, float]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Datasets
    train_dataloader, test_dataloader = make_dataset(cfg_naming['dataset'], device, batch_size=cfg['batch_size'])

    # Model
    model = WraperModel(len(train_dataloader.dataset.classes)).to(device=device)

    # Training params
    max_epochs = cfg["max_epochs"]
    optimizer = make_optimizer(cfg['optimizer'], model, float(cfg['lr']))
    criterion = CrossEntropyLoss()
    max_patience = cfg["patience"]
    patience = 0

    # Best values
    best_accuracy = 0.0
    best_loss = 1e9
    
    # Wandb
    run = wandb.init(
        name = cfg_naming["name"],
        project = cfg_naming["project"],
        config=cfg
    )

    # Main Loop
    with tqdm(range(max_epochs), unit="epochs") as pbar:
        for e in pbar:
            train_mean_loss, train_mean_accuracy, train_samples = train(model, train_dataloader, criterion, optimizer)
            test_mean_loss, test_mean_accuracy, test_samples = test(model, test_dataloader, criterion)

            # Logs
            run.log({
                "train/loss": train_mean_loss,
                "train/accuracy": train_mean_accuracy,
                "test/loss": test_mean_loss,
                "test/accuracy": test_mean_accuracy
            }, step=e+1)

            if e % 5 == 0:
                run.log({
                    "train/samples": train_samples,
                    "test/samples": test_samples
                }, step=e+1)
            
            pbar.set_postfix({"Loss": test_mean_loss, "Acc": test_mean_accuracy})

            # Best Values
            if test_mean_accuracy > best_accuracy:
                best_accuracy = test_mean_accuracy
                torch.save({
                    "epoch": e,
                    "loss": best_loss,
                    "accuracy": best_accuracy,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }, cfg_naming["checkpoint"])

            if test_mean_loss < best_loss:
                best_loss = test_mean_loss
                patience = 0
            else:
                patience += 1

            # Early Stopping
            if patience > max_patience:
                break
        
        # The End :)
        run.finish(0)
        return best_loss, best_accuracy


def main():
    # Process parameters for training (can be a grid search by repeteadly calling main_training() with different configurations)
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str, help="Path to config file.")
    cfg_file = parser.parse_args().cfg
    cfg = parse_config(cfg_file)
    results = []

    best_loss, best_accuracy = main_training(cfg["config"], cfg["names"])

    results.append(cfg["config"] | {"best_loss": best_loss, "best_accuracy": best_accuracy})
    results = {k: [dic[k] for dic in results] for k in results[0]}
    df = pd.DataFrame.from_dict(results, 'columns')
    df.to_csv(cfg["names"]["results_csv"])


if __name__ == "__main__":
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
    