from pathlib import Path
from datetime import datetime
import csv
import torch
import matplotlib.pyplot as plt

class TrainingLogger:
    """Logger for training metrics."""
    def __init__(self, log_dir, lr, dropout, seed):
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
            f.write(f"Learning Rate: {lr}\n")
            f.write(f"Dropout: {dropout}\n")
            f.write(f"Seed: {seed}\n")
            f.write("="*80 + "\n\n")
    
    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, epoch_time):
        """Log metrics for one epoch."""
        # CSV log
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f'{train_loss:.4f}', f'{train_acc:.2f}', 
                           f'{val_loss:.4f}', f'{val_acc:.2f}', f'{epoch_time:.2f}'])
        
        # Text log
        with open(self.log_file, 'a') as f:
            f.write(f"Epoch {epoch}\n")
            f.write(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%\n")
            f.write(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n")
            f.write(f"  Time: {epoch_time:.2f}s\n\n")
    
    def log_final(self, best_epoch, best_val_acc, total_time):
        """Log final results."""
        with open(self.log_file, 'a') as f:
            f.write("\n" + "="*80 + "\n")
            f.write("TRAINING COMPLETED\n")
            f.write("="*80 + "\n")
            f.write(f"Total Training Time: {total_time/60:.2f} minutes\n")
            f.write(f"Best Epoch: {best_epoch}\n")
            f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n")

def visualize_predictions(model, test_loader, device, save_dir, scene_categories, num_samples=16):
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
        pred_label = scene_categories[predictions_all[idx]]
        true_label = scene_categories[labels_all[idx]]
        
        color = 'green' if predictions_all[idx] == labels_all[idx] else 'red'
        ax.set_title(f'Pred: {pred_label}\nGT: {true_label}', 
                    fontsize=10, color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'test_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Test predictions saved to {save_dir / 'test_predictions.png'}")
    