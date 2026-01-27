"""
Training script for the final optimized model.
Uses best hyperparameters from ablation study:
- Learning Rate: 0.001
- Dropout: 0.2
- Optimizer: Adam
- Seed: 42
"""

import argparse
from email import parser
import random
import time
import csv
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import torchvision.models as modelsTorch
import torch.nn.functional as F
import wandb

from model import create_final_model, TeacherModel, create_model_configurable
from data.dataset import load_presplit_dataset, create_data_loaders

# Configuration
SCENE_CATEGORIES = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding']
IMAGE_SIZE = 128
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

LEARNING_RATE = 0.001
DROPOUT = 0.2
BASE_CHANNELS = 32
BATCH_SIZE = 32
EPOCHS = 100
SEED = 42


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(model, teacher, loader, optimizer, device,
                distill=False, alpha=0.5, temperature=4.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()

        student_outputs = model(images)

        if distill:
            with torch.no_grad():
                teacher_outputs = teacher(images)

            loss = distillation_loss(
                student_outputs,
                teacher_outputs,
                labels,
                alpha,
                temperature
            )
        else:
            loss = F.cross_entropy(student_outputs, labels)
        

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = student_outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
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
    
    return total_loss / len(loader), 100. * correct / total


def print_model_summary(model):
    """Print detailed model summary."""
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*80)
    
    total_params = 0
    trainable_params = 0
    
    print(f"\n{'Layer':<30} {'Output Shape':<20} {'Params':<15}")
    print("-"*80)
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += params
            trainable_params += trainable
            
            if params > 0:
                print(f"{name:<30} {'':<20} {params:>12,}")
    
    print("-"*80)
    print(f"{'Total Parameters:':<50} {total_params:>12,}")
    print(f"{'Trainable Parameters:':<50} {trainable_params:>12,}")
    print(f"{'Non-trainable Parameters:':<50} {total_params - trainable_params:>12,}")
    print("="*80 + "\n")


class TrainingLogger:
    """Logger for training metrics."""
    def __init__(self, log_dir, model=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f'training_log_{timestamp}.txt'
        self.csv_file = self.log_dir / f'training_metrics_{timestamp}.csv'
        
        # Initialize CSV
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'LR', 'Time (s)'])
        
        # Initialize text log with model summary
        with open(self.log_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TRAINING LOG - FINAL OPTIMIZED MODEL\n")
            f.write("="*80 + "\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: SE_R4_6Layer\n")
            f.write(f"Learning Rate: {LEARNING_RATE}\n")
            f.write(f"Dropout: {DROPOUT}\n")
            f.write(f"Seed: {SEED}\n")
            f.write("="*80 + "\n\n")
            
            # Add model summary if provided
            if model is not None:
                f.write("="*80 + "\n")
                f.write("MODEL ARCHITECTURE SUMMARY\n")
                f.write("="*80 + "\n\n")
                
                total_params = 0
                trainable_params = 0
                
                f.write(f"{'Layer':<30} {'Output Shape':<20} {'Params':<15}\n")
                f.write("-"*80 + "\n")
                
                for name, module in model.named_modules():
                    if len(list(module.children())) == 0:  # Leaf modules only
                        params = sum(p.numel() for p in module.parameters())
                        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                        total_params += params
                        trainable_params += trainable
                        
                        if params > 0:
                            f.write(f"{name:<30} {'':<20} {params:>12,}\n")
                
                f.write("-"*80 + "\n")
                f.write(f"{'Total Parameters:':<50} {total_params:>12,}\n")
                f.write(f"{'Trainable Parameters:':<50} {trainable_params:>12,}\n")
                f.write(f"{'Non-trainable Parameters:':<50} {total_params - trainable_params:>12,}\n")
                f.write("="*80 + "\n\n")
    
    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, lr, epoch_time):
        """Log metrics for one epoch."""
        # CSV log
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f'{train_loss:.4f}', f'{train_acc:.2f}', 
                           f'{val_loss:.4f}', f'{val_acc:.2f}', f'{lr:.6f}', f'{epoch_time:.2f}'])
        
        # Text log
        with open(self.log_file, 'a') as f:
            f.write(f"Epoch {epoch}\n")
            f.write(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%\n")
            f.write(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n")
            f.write(f"  Learning Rate: {lr:.6f}\n")
            f.write(f"  Time: {epoch_time:.2f}s\n\n")
    
    def log_final(self, best_epoch, best_val_acc, test_acc, total_time):
        """Log final results."""
        with open(self.log_file, 'a') as f:
            f.write("\n" + "="*80 + "\n")
            f.write("TRAINING COMPLETED\n")
            f.write("="*80 + "\n")
            f.write(f"Total Training Time: {total_time/60:.2f} minutes\n")
            f.write(f"Best Epoch: {best_epoch}\n")
            f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
            f.write(f"Test Accuracy: {test_acc:.2f}%\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n")


def plot_training_curves(csv_file, save_dir):
    """Plot training and validation curves from CSV log."""
    import pandas as pd
    
    # Read CSV
    df = pd.read_csv(csv_file)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Loss curves
    axes[0, 0].plot(df['Epoch'], df['Train Loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(df['Epoch'], df['Val Loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy curves
    axes[0, 1].plot(df['Epoch'], df['Train Acc'], label='Train Acc', linewidth=2)
    axes[0, 1].plot(df['Epoch'], df['Val Acc'], label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Learning rate
    axes[1, 0].plot(df['Epoch'], df['LR'], linewidth=2, color='green')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Plot 4: Epoch time
    axes[1, 1].plot(df['Epoch'], df['Time (s)'], linewidth=2, color='orange')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Time (seconds)', fontsize=12)
    axes[1, 1].set_title('Training Time per Epoch', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {save_dir / 'training_curves.png'}")


def visualize_predictions(model, test_loader, device, save_dir, num_samples=16):
    """Visualize sample predictions from test set."""
    model.eval()
    
    images_list = []
    predictions_list = []
    labels_list = []
    
    # Collect samples
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            images_list.append(images.cpu())
            predictions_list.append(predicted.cpu())
            labels_list.append(labels.cpu())
            
            if len(images_list) * images.size(0) >= num_samples:
                break
    
    # Concatenate
    images_all = torch.cat(images_list)[:num_samples]
    predictions_all = torch.cat(predictions_list)[:num_samples]
    labels_all = torch.cat(labels_list)[:num_samples]
    
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images_all = images_all * std + mean
    images_all = torch.clamp(images_all, 0, 1)
    
    # Plot
    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
    
    for idx in range(num_samples):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Convert to numpy and transpose
        img = images_all[idx].permute(1, 2, 0).numpy()
        
        # Display image
        ax.imshow(img)
        ax.axis('off')
        
        # Add prediction and ground truth
        pred_label = SCENE_CATEGORIES[predictions_all[idx]]
        true_label = SCENE_CATEGORIES[labels_all[idx]]
        
        color = 'green' if predictions_all[idx] == labels_all[idx] else 'red'
        ax.set_title(f'Pred: {pred_label}\nGT: {true_label}', 
                    fontsize=10, color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'test_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Test predictions saved to {save_dir / 'test_predictions.png'}")


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


def get_curated_config():

    base_config = {
        'lr': LEARNING_RATE,
        'dropout_rate': DROPOUT,
        'base_channels': BASE_CHANNELS,
        'seed': SEED
    }

    def get_channels(base, num_layers):
        channels = [base]
        for i in range(1, num_layers):
            channels.append(min(base * (2 ** (i // 2)), 256))
        return channels

    config = {
        'name': 'baseline_relu_ln',
        'num_conv_layers': 5,
        'channels': get_channels(BASE_CHANNELS, 5),
        'activation': 'relu',
        'normalization': 'layernorm',
        'component_order': 'conv_norm_act',
        'use_residual': False,
        'use_attention': False,
        'use_se_block': False,
        'use_recurrent': False,
        **base_config
    }

    return config



def main():
    parser = argparse.ArgumentParser(description='Train final optimized model')
    parser.add_argument('--data-dir', type=str, default='dataset/MIT_split', help='Dataset directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Directory to save model')
    parser.add_argument('--multi-gpu', action='store_true', help='Use multi-GPU training')
    parser.add_argument('--distill', action='store_true', help='Enable knowledge distillation using InceptionV3 teacher')
    parser.add_argument('--kd-alpha', type=float, default=0.5, help='Weight for CE loss (1-alpha for KD loss)')
    parser.add_argument('--kd-temperature', type=float, default=4.0, help='Temperature for distillation')
    parser.add_argument('--finetuned_teacher', action='store_true', help='Use finetuned teacher model. Only used if --distill is set.')
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(SEED)
    
    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'-'*80}")
    print("TRAINING FINAL OPTIMIZED MODEL")
    print(f"{'-'*80}")
    print(f"Model: SE_R4_6Layer (Best from ablation study)")
    print(f"Expected Validation Accuracy: 91.52%")
    print(f"Expected Test Accuracy: 90.83%")
    print(f"Parameters: ~2.1M")
    print(f"\nHyperparameters:")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Dropout: {DROPOUT}")
    print(f"  Optimizer: Adam")
    print(f"  Seed: {SEED}")
    if args.multi_gpu and torch.cuda.device_count() > 1:
        print(f"  Multi-GPU: Enabled ({torch.cuda.device_count()} GPUs)")
    print(f"{'-'*80}\n")
    
    # Load data
    data_dir = Path(args.data_dir)
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = load_presplit_dataset(
        data_dir, SCENE_CATEGORIES, seed=SEED
    )
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels,
        batch_size=args.batch_size, num_workers=4, image_size=IMAGE_SIZE, mean=MEAN, std=STD
    )
    
    print(f"Dataset loaded:")
    print(f"  Train: {len(train_paths)} images")
    print(f"  Val: {len(val_paths)} images")
    print(f"  Test: {len(test_paths)} images\n")
    
    # Create model
    #model = create_final_model(num_classes=len(SCENE_CATEGORIES))
    config = get_curated_config()
    model = create_model_configurable('configurable_cnn', num_classes=len(SCENE_CATEGORIES), **config)

    # Knowledge Distillation setup
    teacher_model = None
    if args.distill:
        print("Knowledge distillation ENABLED (Teacher: InceptionV3)")
        teacher_model = TeacherModel(num_classes=8, pretrained=True)        

        if args.finetuned_teacher:    
            print("Loading finetuned teacher model weights...")
            checkpoint = torch.load("/home/gasbert/master/C3/project-team-3/Week3/checkpoints/best_model_fold_0.pth", map_location="cpu", weights_only=False)
            teacher_model.load_state_dict(checkpoint["model_state_dict"], strict=True)

        

        if args.multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            teacher_model = nn.DataParallel(teacher_model)
            
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        
        for p in teacher_model.parameters():
            p.requires_grad = False
    
    # Multi-GPU support
    if args.multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    num_params = model.module.get_num_parameters() if isinstance(model, nn.DataParallel) else model.get_num_parameters()

    print("\nEvaluating teacher model on validation set...")
    teacher_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = teacher_model(images)
            preds = outputs.argmax(dim=1)
            print("PREDS")
            print(preds[:20])
            print("LABELS")
            print(labels[:20])
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print("Teacher val acc:", 100 * correct / total)





    wandb.init(
        project="scene-classification",   # change if you want
        name=f"SE_R4_6Layer{'_KD' if args.distill else ''}",
        config={
            "model": "SE_R4_6Layer",
            "num_parameters": num_params,
            "dataset": "MIT_split",
            "image_size": IMAGE_SIZE,
            "batch_size": args.batch_size,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "dropout": DROPOUT,
            "optimizer": "Adam",
            "seed": SEED,
            "distillation": args.distill,
            "teacher_model": "InceptionV3" if args.distill else None,
            "finetuned_teacher": args.finetuned_teacher if args.distill else None,
        }
    )

    wandb.watch(
        model,
        log="gradients",
        log_freq=100
    )

    if args.distill:
        kd_alpha = wandb.config.kd_alpha
        kd_temperature = wandb.config.kd_temperature
        print(f"Using KD alpha: {kd_alpha}, temperature: {kd_temperature}")
    else:
        kd_alpha = None
        kd_temperature = None
    
    # Print model summary
    print_model_summary(model.module if isinstance(model, nn.DataParallel) else model)
    
    print(f"Total model parameters: {num_params:,}\n")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Initialize logger
    log_dir = Path('training_logs')
    logger = TrainingLogger(log_dir)
    print(f"Training logs will be saved to: {log_dir}/\n")
    
    # Training loop
    best_val_acc = 0
    best_epoch = 0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    print("Starting training...\n")
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        
        print(f"Epoch {epoch}/{EPOCHS}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc = train_epoch(model, teacher_model, train_loader, optimizer, device,
                                           distill=args.distill,
                                           alpha=kd_alpha,
                                           temperature=kd_temperature)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']

        if new_lr != current_lr:
            print(f"Learning rate reduced from {current_lr:.6f} to {new_lr:.6f}")
        
        epoch_time = time.time() - epoch_start
        
        # Log metrics
        logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, current_lr, epoch_time)

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "learning_rate": current_lr,
            "epoch_time_sec": epoch_time,
        })
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, save_dir / 'best_model.pth')

            wandb.log({
                "best/val_accuracy": best_val_acc,
                "best/epoch": best_epoch,
            })
            
            print(f"New best model saved! (Val Acc: {val_acc:.2f}%)")
        
        print(f"Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch})")
        print()
    
    training_time = time.time() - start_time
    
    # Load best model and evaluate on test set
    print(f"\n{'-'*80}")
    print("FINAL EVALUATION")
    print(f"{'-'*80}\n")
    
    checkpoint = torch.load(save_dir / 'best_model.pth')
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    
    # Log final results
    logger.log_final(best_epoch, best_val_acc, test_acc, training_time)
    
    # Generate plots
    print(f"\n{'-'*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'-'*80}\n")
    
    # Plot training curves
    plot_training_curves(logger.csv_file, log_dir)
    
    # Visualize test predictions
    visualize_predictions(model, test_loader, device, log_dir, num_samples=16)
    
    print(f"\n{'-'*80}")
    print("TRAINING COMPLETE")
    print(f"{'-'*80}")
    print(f"Training completed in {training_time/60:.2f} minutes")
    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"{'-'*80}\n")

    wandb.finish()


if __name__ == "__main__":
    main()
