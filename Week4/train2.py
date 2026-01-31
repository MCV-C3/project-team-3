"""
Training script for the final optimized model.
Uses best hyperparameters from ablation study:
- Learning Rate: 0.001
- Dropout: 0.2
- Optimizer: Adam
- Seed: 42
"""

import argparse
import random
import time
from pathlib import Path, PurePath
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import torchvision.models as modelsTorch
import torch.nn.functional as F
import wandb
import os

from data.dataset2 import create_data_loaders
from model2 import build_model
from logs2 import TrainingLogger, visualize_predictions

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_arguments():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model', type=str, default='dummy_test', help='Model for experiment')

    # Dataset
    parser.add_argument('--data-dir', type=str, default='dataset/MIT_small_train1', help='Dataset directory')

    # Hyperparameters
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Max number of epochs')
    parser.add_argument('--estopping', type=int, default=5, help='Early stopping patience')
    
    # Distillation
    parser.add_argument('--distill', action='store_true', help='Enable knowledge distillation using InceptionV3 teacher')
    parser.add_argument('--kd-alpha', type=float, default=0.5, help='Weight for CE loss (1-alpha for KD loss)')
    parser.add_argument('--kd-temperature', type=float, default=4.0, help='Temperature for distillation')
    parser.add_argument('--finetuned_teacher', type=str, default=None, help='Path to weights of finetuned teacher, if any. Only used if --distill is set.')

    # Extras
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory to save model')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Directory to save model')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--multi-gpu', action='store_true', help='Use multi-GPU training')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')

    args = parser.parse_args()
    return args

def distillation_loss(student_logits, teacher_logits, labels,
                      alpha=0.5, temperature=4.0):
    """
    Compute combined CE + KD loss.
    """
    ce_loss = F.cross_entropy(student_logits, labels)

    kd_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction='batchmean'
    )

    return alpha * ce_loss + (1 - alpha) * (temperature ** 2) * kd_loss

def train_epoch(model, loader, optimizer, device,
                teacher=None, distill=False, alpha=0.5, temperature=4.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()

        outputs = model(images)

        if distill:
            with torch.no_grad():
                teacher_outputs = teacher(images)

            loss = distillation_loss(
                outputs,
                teacher_outputs,
                labels,
                alpha,
                temperature
            )
        else:
            loss = F.cross_entropy(outputs, labels)
        

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    return total_loss / len(loader), correct / total

@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Validating', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return total_loss / len(loader), correct / total

def main():
    args = parse_arguments()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Initialize wandb
    wandb.init(
        project="MCVC-C3-W4",
        name=f"{args.model}{'_KD' if args.distill else ''}",
        config={
            "model": f"{args.model}",
            "dataset": PurePath(args.data_dir),
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "dropout": args.dropout,
            "optimizer": "Adam",
            "seed": args.seed,
            "distillation": args.distill,
            "kd_alpha": args.kd_alpha if args.distill else None,
            "kd_temperature": args.kd_temperature if args.distill else None,
            "teacher_model": "InceptionV3" if args.distill else None,
            "finetuned_teacher": args.finetuned_teacher if args.distill else None,
        }
    )
    
    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.multi_gpu and torch.cuda.device_count() > 1:
        print(f"  Multi-GPU: Enabled ({torch.cuda.device_count()} GPUs)")
    print(f"{'-'*80}\n")

    # Load data
    data_dir = Path(args.data_dir)
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'test'
    
    train_loader, val_loader = create_data_loaders(
        train_dir, val_dir, device,
        batch_size=args.batch_size, num_workers=0
    )
    
    print(f"Dataset loaded:")
    print(f"  Train: {len(train_loader.dataset)} images")
    print(f"  Val: {len(val_loader.dataset)} images")

    # Create model
    model_params = {
        'dropout': args.dropout,
        'depth': wandb.config.depth,
        'num_classes': len(train_loader.dataset.classes)
    }
    model = build_model(args.model, **model_params)

    if args.multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)
    num_params = model.module.get_num_parameters() if isinstance(model, nn.DataParallel) else model.get_num_parameters()
    wandb.log({"num_parameters": num_params})

    # Knowledge Distillation Setup
    teacher_model = None
    if args.distill:
        print("Knowledge distillation ENABLED (Teacher: InceptionV3)")
        teacher_model = build_model('teacher_model', num_classes=8, pretrained=True)        

        if args.finetuned_teacher:    
            print("Loading finetuned teacher model weights...")
            checkpoint = torch.load(args.finetuned_teacher, map_location="cpu", weights_only=False)
            teacher_model.load_state_dict(checkpoint["model_state_dict"], strict=True)        

        if args.multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            teacher_model = nn.DataParallel(teacher_model)
            
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    wandb.watch(
        model,
        log="gradients",
        log_freq=100
    )

    wandb.config['num_params'] = num_params

    print(f"Total model parameters: {num_params:,}\n")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Initialize logger
    log_dir = Path(args.log_dir)
    logger = TrainingLogger(log_dir, args.lr, args.dropout, args.seed)

    # Training loop
    best_val_acc = 0
    best_epoch = 0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    patience = 0

    print("Starting training...\n")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        patience += 1
        epoch_start = time.time()
        
        print(f"Epoch {epoch}/{args.epochs}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device,
                                            teacher=teacher_model,
                                            distill=args.distill,
                                            alpha=args.kd_alpha,
                                            temperature=args.kd_temperature)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        
        # Log metrics
        logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, epoch_time)

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "epoch_time_sec": epoch_time,
        })
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Time: {epoch_time:.2f}s")
        

        # Save best model
        if val_acc > best_val_acc:
            patience = 0
            best_val_acc = val_acc
            best_epoch = epoch

            # ---- Derived efficiency metrics ----
            params_100k = num_params / 100_000
            efficiency_ratio = val_acc / params_100k
            distance_d = np.sqrt(
                (params_100k ** 2) +
                ((val_acc * 100 if val_acc <= 1 else val_acc) - 100) ** 2
            )
            
            os.makedirs(save_dir / args.model, exist_ok=True)
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, save_dir / args.model / 'best_model.pth')

            wandb.log({
                "best/val_accuracy": best_val_acc,
                "best/epoch": best_epoch,
                "best/efficiency_ratio": efficiency_ratio,
                "best/distance_d": distance_d,
            })
            
            print(f"New best model saved! (Val Acc: {val_acc:.2f}%)")
        
        print(f"Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch})\n")

        if patience > args.estopping:
            break
    
    training_time = time.time() - start_time

    # Log final results
    logger.log_final(best_epoch, best_val_acc, training_time)

    os.makedirs(save_dir / args.model, exist_ok=True)
    checkpoint = torch.load(save_dir / args.model / 'best_model.pth')
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    # Visualize test predictions
    visualize_predictions(model, val_loader, device, log_dir, val_loader.dataset.classes, num_samples=16)

    wandb.finish()

if __name__ == '__main__':
    main()