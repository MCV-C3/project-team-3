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
import torchvision.transforms.v2 as F
import tqdm
import argparse
import pickle
import os
from bovw import BOVW, visualize_bow_histogram
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Optional: torchviz for computational graph visualization
try:
    from torchviz import make_dot
    TORCHVIZ_AVAILABLE = True
except ImportError:
    TORCHVIZ_AVAILABLE = False
    print("Note: torchviz not available. Computational graph will not be generated.")
    print("Install with: pip install torchviz")

PATCH_SIZE = 8

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


def extract_mlp_features(model, dataloader, device, layer_name='hidden'):
    """Extract features from a specific layer of the MLP for all patches"""
    model.eval()
    all_features = []
    all_labels = []
    
    # Register hook to extract features
    features_dict = {}
    
    def get_activation(name):
        def hook(model, input, output):
            features_dict[name] = output.detach()
        return hook
    
    # Register hook on the desired layer based on your model structure
    if layer_name == 'hidden':
        # Extract from layer2 (after second hidden layer, before output)
        handle = model.layer2.register_forward_hook(get_activation('hidden'))
    elif layer_name == 'output':
        # Extract from output_layer
        handle = model.output_layer.register_forward_hook(get_activation('output'))
    else:
        raise ValueError(f"Unknown layer: {layer_name}")
    
    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(dataloader, desc=f"Extracting {layer_name} features"):
            inputs = inputs.to(device)
            
            batch_size, C, H, W = inputs.shape
            num_patches_h = H // PATCH_SIZE
            num_patches_w = W // PATCH_SIZE
            num_patches = num_patches_h * num_patches_w

            # Convert to patches
            patches = inputs.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
            patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
            patches = patches.view(batch_size, num_patches, C, PATCH_SIZE, PATCH_SIZE)
            patches = patches.view(batch_size * num_patches, C*PATCH_SIZE*PATCH_SIZE)
            
            # Forward pass to trigger hook
            _ = model(patches)
            
            # Get features for this batch
            features = features_dict[layer_name].cpu().numpy()
            
            # Reshape to [batch_size, num_patches, feature_dim]
            feature_dim = features.shape[1]
            features = features.reshape(batch_size, num_patches, feature_dim)
            
            all_features.append(features)
            all_labels.extend(labels.cpu().numpy())
    
    handle.remove()
    
    # Concatenate all features: [num_images, num_patches, feature_dim]
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.array(all_labels)
    
    return all_features, all_labels


def train_bovw_classifier(bovw_model, train_features, train_labels, test_features, test_labels):
    """Train BoVW using MLP features as dense descriptors"""
    
    print("Building BoVW codebook from training features...")
    # Flatten all patches from all images into one big array
    # train_features shape: [num_images, num_patches, feature_dim]
    # We want: [num_images * num_patches, feature_dim]
    num_images, num_patches, feature_dim = train_features.shape
    train_descriptors_flat = train_features.reshape(-1, feature_dim)
    
    print(f"Total descriptors for codebook: {train_descriptors_flat.shape[0]} x {train_descriptors_flat.shape[1]}")
    
    # Fit codebook on all patch descriptors (as a single array, not a list)
    bovw_model.fit_codebook([train_descriptors_flat])
    
    print("Computing BoVW histograms for training set...")
    train_histograms = []
    for i in tqdm.tqdm(range(len(train_features)), desc="Train BoVW histograms"):
        # Each image's patches: [num_patches, feature_dim]
        hist = bovw_model.compute_histogram(train_features[i])
        train_histograms.append(hist)
    train_histograms = np.array(train_histograms)
    
    print("Computing BoVW histograms for test set...")
    test_histograms = []
    for i in tqdm.tqdm(range(len(test_features)), desc="Test BoVW histograms"):
        hist = bovw_model.compute_histogram(test_features[i])
        test_histograms.append(hist)
    test_histograms = np.array(test_histograms)
    
    # Visualize a few histograms
    os.makedirs("./bovw_visualizations", exist_ok=True)
    for i in range(min(3, len(test_histograms))):
        visualize_bow_histogram(test_histograms[i], i, output_folder="./bovw_visualizations")
    
    # Train SVM classifier on BoVW histograms
    print("Training SVM classifier on BoVW histograms...")
    svm_classifier = LinearSVC(max_iter=5000, random_state=42)
    svm_classifier.fit(train_histograms, train_labels)
    
    # Evaluate
    train_pred = svm_classifier.predict(train_histograms)
    test_pred = svm_classifier.predict(test_histograms)
    
    train_acc = accuracy_score(train_labels, train_pred)
    test_acc = accuracy_score(test_labels, test_pred)
    
    return train_acc, test_acc, svm_classifier, train_pred, test_pred


def train_fisher_vector_classifier(bovw_model, train_features, train_labels, test_features, test_labels):
    """Train using Fisher Vectors (GMM-based) instead of histograms"""
    
    print("Building GMM codebook from training features...")
    num_images, num_patches, feature_dim = train_features.shape
    train_descriptors_flat = train_features.reshape(-1, feature_dim)
    
    print(f"Total descriptors: {train_descriptors_flat.shape[0]} x {train_descriptors_flat.shape[1]}")
    
    # Sample for GMM (Fisher Vectors are memory intensive)
    # Use more samples for better codebook quality
    sample_size = min(200000, len(train_descriptors_flat))
    print(f"Sampling {sample_size:,} descriptors for GMM training...")
    sampled_idx = np.random.choice(len(train_descriptors_flat), sample_size, replace=False)
    sampled_descriptors = train_descriptors_flat[sampled_idx]
    
    # Fit GMM codebook
    bovw_model.fit_codebook([sampled_descriptors])
    
    print("Computing Fisher Vectors for training set...")
    train_fisher_vecs = []
    for i in tqdm.tqdm(range(len(train_features)), desc="Train Fisher Vectors"):
        fv = bovw_model.compute_fisher_vector(train_features[i])
        train_fisher_vecs.append(fv)
    train_fisher_vecs = np.array(train_fisher_vecs)
    
    print(f"Fisher Vector shape: {train_fisher_vecs.shape}")
    
    print("Computing Fisher Vectors for test set...")
    test_fisher_vecs = []
    for i in tqdm.tqdm(range(len(test_features)), desc="Test Fisher Vectors"):
        fv = bovw_model.compute_fisher_vector(test_features[i])
        test_fisher_vecs.append(fv)
    test_fisher_vecs = np.array(test_fisher_vecs)
    
    # Train SVM classifier on Fisher Vectors
    print("Training SVM classifier on Fisher Vectors...")
    svm_classifier = LinearSVC(max_iter=5000, random_state=42, dual='auto')
    svm_classifier.fit(train_fisher_vecs, train_labels)
    
    # Evaluate
    train_pred = svm_classifier.predict(train_fisher_vecs)
    test_pred = svm_classifier.predict(test_fisher_vecs)
    
    train_acc = accuracy_score(train_labels, train_pred)
    test_acc = accuracy_score(test_labels, test_pred)
    
    return train_acc, test_acc, svm_classifier, train_pred, test_pred


def plot_metrics(train_metrics: Dict, test_metrics: Dict, metric_name: str, save_prefix: str = ""):
    """
    Plots and saves metrics for training and testing.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics[metric_name], label=f'Train {metric_name.capitalize()}')
    plt.plot(test_metrics[metric_name], label=f'Test {metric_name.capitalize()}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.capitalize())
    plt.title(f'{metric_name.capitalize()} Over Epochs')
    plt.legend()
    plt.grid(True)

    filename = f"{save_prefix}{metric_name}.png"
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    plt.close()


def plot_comparison(end2end_results: Dict, bovw_results: Dict, save_path: str = "comparison.png"):
    """Plot comparison between end-to-end and BoVW approaches"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy comparison - use best test accuracy for end-to-end
    methods = [
        'End-to-End\n(Mean Agg)', 
        'BoVW Hidden\n(K-means)', 
        'BoVW Hidden\n(Fisher)',
        'BoVW Output\n(K-means)', 
        'BoVW Output\n(Fisher)'
    ]
    train_accs = [
        end2end_results['train_acc'], 
        bovw_results['hidden_kmeans']['train_acc'],
        bovw_results['hidden_fisher']['train_acc'],
        bovw_results['output_kmeans']['train_acc'],
        bovw_results['output_fisher']['train_acc']
    ]
    test_accs = [
        end2end_results['best_test_acc'], 
        bovw_results['hidden_kmeans']['test_acc'],
        bovw_results['hidden_fisher']['test_acc'],
        bovw_results['output_kmeans']['test_acc'],
        bovw_results['output_fisher']['test_acc']
    ]
    
    x = np.arange(len(methods))
    width = 0.35
    
    axes[0].bar(x - width/2, train_accs, width, label='Train', alpha=0.8, color='steelblue')
    axes[0].bar(x + width/2, test_accs, width, label='Test (Best)', alpha=0.8, color='coral')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Accuracy Comparison: End-to-End vs BoVW Methods', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, fontsize=9)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim([0, 1])
    
    # Add value labels on bars
    for i, (train_acc, test_acc) in enumerate(zip(train_accs, test_accs)):
        axes[0].text(i - width/2, train_acc + 0.02, f'{train_acc:.3f}', 
                    ha='center', va='bottom', fontsize=8)
        axes[0].text(i + width/2, test_acc + 0.02, f'{test_acc:.3f}', 
                    ha='center', va='bottom', fontsize=8)
    
    # Training curves for end-to-end with best accuracy marker
    if end2end_results.get('train_accuracies') and len(end2end_results['train_accuracies']) > 0:
        axes[1].plot(end2end_results['train_accuracies'], label='Train Acc', linewidth=2)
        axes[1].plot(end2end_results['test_accuracies'], label='Test Acc', linewidth=2)
        
        # Mark best test accuracy
        best_epoch = np.argmax(end2end_results['test_accuracies'])
        best_acc = end2end_results['best_test_acc']
        axes[1].scatter([best_epoch], [best_acc], color='red', s=100, zorder=5, 
                       label=f'Best: {best_acc:.3f}')
        axes[1].axhline(y=best_acc, color='red', linestyle='--', alpha=0.3)
        
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Accuracy', fontsize=11)
        axes[1].set_title('End-to-End Training Progress', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved as {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title="Confusion Matrix"):
    """
    Plot and save confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
        title: Title for the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title(f'{title} (Counts)')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Plot normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Proportion'})
    axes[1].set_title(f'{title} (Normalized)')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved as {save_path}")
    plt.close()


def get_predictions(model, dataloader, device):
    """Get all predictions and true labels from a dataloader"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(dataloader, desc="Getting predictions"):
            inputs, labels = inputs.to(device), labels.to(device)

            batch_size, C, H, W = inputs.shape
            num_patches_h = H // PATCH_SIZE
            num_patches_w = W // PATCH_SIZE
            num_patches = num_patches_h * num_patches_w

            # Convert to patches
            patches = inputs.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
            patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
            patches = patches.view(batch_size, num_patches, C, PATCH_SIZE, PATCH_SIZE)
            patches = patches.view(batch_size * num_patches, C*PATCH_SIZE*PATCH_SIZE)
            
            # Forward pass
            patch_outputs = model(patches)
            patch_outputs = patch_outputs.view(batch_size, num_patches, -1)
            outputs = patch_outputs.mean(dim=1)
            
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds)


def plot_computational_graph(model: torch.nn.Module, input_size: tuple, device: torch.device, filename: str = "computational_graph"):
    """
    Generates and saves a plot of the computational graph of the model.
    """
    if not TORCHVIZ_AVAILABLE:
        print("Skipping computational graph (torchviz not installed)")
        return
    
    try:
        model.eval()
        dummy_input = torch.randn(*input_size).to(device)
        graph = make_dot(model(dummy_input), params=dict(model.named_parameters()), 
                         show_attrs=True).render(filename, format="png")
        print(f"Computational graph saved as {filename}")
    except Exception as e:
        print(f"Warning: Could not generate computational graph: {e}")
        print("(This is optional and doesn't affect the training)")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train and compare MLP approaches")
    parser.add_argument("--skip_training", action="store_true", 
                       help="Skip end-to-end training and load saved model")
    parser.add_argument("--model_path", type=str, default="best_patch_model.pth",
                       help="Path to saved model")
    parser.add_argument("--codebook_size", type=int, default=128,
                       help="Size of BoVW codebook")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    # Data loading
    transformation_t = F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.Resize(size=(64, 64)),
        F.RandomHorizontalFlip(),
        F.RandomRotation(20),
    ])

    transformation = F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.Resize(size=(64, 64)),
    ])
    
    data_train = ImageFolder("/data2/users/gasbert/master/C3/2425/MIT_small_train_1/train", 
                            transform=transformation_t)
    data_test = ImageFolder("/data2/users/gasbert/master/C3/2425/MIT_small_train_1/test", 
                           transform=transformation) 

    train_loader = DataLoader(data_train, batch_size=256, pin_memory=True, 
                             shuffle=True, num_workers=8)
    test_loader = DataLoader(data_test, batch_size=128, pin_memory=True, 
                            shuffle=False, num_workers=8)

    C, H, W = np.array(data_train[0][0]).shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")
    
    # Initialize model
    model = SimpleModel(input_d=C*PATCH_SIZE*PATCH_SIZE, hidden_d=300, output_d=11)
    model = model.to(device)
    
    end2end_results = {}
    
    # ==================== PART 1: End-to-End Training ====================
    if not args.skip_training:
        print("\n" + "="*60)
        print("PART 1: Training End-to-End Patch-Based MLP")
        print("="*60 + "\n")
        
        plot_computational_graph(model, input_size=(1, C*PATCH_SIZE*PATCH_SIZE), device=device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 120

        train_losses, train_accuracies = [], []
        test_losses, test_accuracies = [], []
        best_test_acc = 0.0
        best_epoch = 0
        
        for epoch in tqdm.tqdm(range(num_epochs), desc="Training End-to-End"):
            train_loss, train_accuracy = train_by_patches(model, train_loader, criterion, 
                                                         optimizer, device)
            test_loss, test_accuracy = test_by_patches(model, test_loader, criterion, device)

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            print(f"Epoch {epoch + 1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")
            
            # Save best model
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                best_epoch = epoch
                torch.save(model.state_dict(), args.model_path)
                print(f"  -> Best model saved with test accuracy: {best_test_acc:.4f}")

        # Plot end-to-end results
        plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, 
                    {"loss": test_losses, "accuracy": test_accuracies}, 
                    "loss", save_prefix="end2end_")
        plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, 
                    {"loss": test_losses, "accuracy": test_accuracies}, 
                    "accuracy", save_prefix="end2end_")
        
        # Load best model for confusion matrix
        model.load_state_dict(torch.load(args.model_path))
        
        # Get predictions for confusion matrix
        print("\nGenerating confusion matrix for best model...")
        train_true, train_pred = get_predictions(model, train_loader, device)
        test_true, test_pred = get_predictions(model, test_loader, device)
        
        # Get class names
        class_names = [str(i) for i in range(11)]
        if hasattr(data_train, 'classes'):
            class_names = data_train.classes
        
        # Plot confusion matrices
        plot_confusion_matrix(train_true, train_pred, class_names,
                            "end2end_train_confusion_matrix.png",
                            "End-to-End Train Confusion Matrix")
        plot_confusion_matrix(test_true, test_pred, class_names,
                            "end2end_test_confusion_matrix.png",
                            "End-to-End Test Confusion Matrix")
        
        # Calculate train accuracy at best epoch
        best_train_acc = train_accuracies[best_epoch]
        
        end2end_results = {
            'train_acc': best_train_acc,
            'test_acc': best_test_acc,
            'best_test_acc': best_test_acc,
            'best_epoch': best_epoch,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
            'train_losses': train_losses,
            'test_losses': test_losses
        }
        
        print(f"\nEnd-to-End Results:")
        print(f"  Best Epoch: {best_epoch + 1}")
        print(f"  Train Accuracy (at best epoch): {best_train_acc:.4f}")
        print(f"  Best Test Accuracy: {best_test_acc:.4f}")
        print(f"  Final Train Accuracy: {train_accuracies[-1]:.4f}")
        print(f"  Final Test Accuracy: {test_accuracies[-1]:.4f}")
        
    else:
        print("\n" + "="*60)
        print("Loading pre-trained model...")
        print("="*60 + "\n")
        model.load_state_dict(torch.load(args.model_path))
        
        # Quick evaluation
        criterion = nn.CrossEntropyLoss()
        _, train_acc = test_by_patches(model, train_loader, criterion, device)
        _, test_acc = test_by_patches(model, test_loader, criterion, device)
        
        # Get predictions for confusion matrix
        print("\nGenerating confusion matrix...")
        train_true, train_pred = get_predictions(model, train_loader, device)
        test_true, test_pred = get_predictions(model, test_loader, device)
        
        # Get class names
        class_names = [str(i) for i in range(11)]
        if hasattr(data_train, 'classes'):
            class_names = data_train.classes
        
        # Plot confusion matrices
        plot_confusion_matrix(train_true, train_pred, class_names,
                            "end2end_train_confusion_matrix.png",
                            "End-to-End Train Confusion Matrix")
        plot_confusion_matrix(test_true, test_pred, class_names,
                            "end2end_test_confusion_matrix.png",
                            "End-to-End Test Confusion Matrix")
        
        end2end_results = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'best_test_acc': test_acc,
            'train_accuracies': [],
            'test_accuracies': []
        }
        
        print(f"Loaded model performance:")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
    
    # ==================== PART 2: BoVW with MLP Features ====================
    print("\n" + "="*60)
    print("PART 2: BoVW with MLP Features as Dense Descriptors")
    print("="*60 + "\n")
    
    # Load best model for feature extraction
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    bovw_results = {
        'hidden_kmeans': {}, 
        'hidden_fisher': {},
        'output_kmeans': {}, 
        'output_fisher': {}
    }
    
    # Test both hidden layer and output layer features
    for layer_name in ['hidden', 'output']:
        print(f"\n{'='*60}")
        print(f"Processing {layer_name.upper()} layer features")
        print(f"{'='*60}")
        
        # Extract features
        train_features, train_labels = extract_mlp_features(model, train_loader, device, 
                                                            layer_name=layer_name)
        test_features, test_labels = extract_mlp_features(model, test_loader, device, 
                                                          layer_name=layer_name)
        
        print(f"Train features shape: {train_features.shape}")
        print(f"Test features shape: {test_features.shape}")
        
        # Get class names
        class_names = [str(i) for i in range(11)]
        if hasattr(data_train, 'classes'):
            class_names = data_train.classes
        
        # Test both K-means (BoVW histogram) and GMM (Fisher Vector)
        for method in ['kmeans', 'fisher']:
            print(f"\n--- Using {layer_name} layer with {method.upper()} ---")
            
            if method == 'kmeans':
                # K-means BoVW with histograms
                codebook_size = args.codebook_size
                bovw_model = BOVW(
                    descriptor_type='precomputed',
                    codebook_size=codebook_size,
                    codebook_type='kmeans'
                )
                
                train_acc, test_acc, svm, train_pred, test_pred = train_bovw_classifier(
                    bovw_model, train_features, train_labels, test_features, test_labels
                )
                
            else:  # fisher
                # GMM-based Fisher Vectors
                # Use smaller codebook for Fisher Vectors (they're much higher dimensional)
                codebook_size = 128 if layer_name == 'hidden' else 16
                print(f"Using codebook size: {codebook_size} (Fisher Vectors)")
                
                bovw_model = BOVW(
                    descriptor_type='precomputed',
                    codebook_size=codebook_size,
                    codebook_type='gmm'
                )
                
                train_acc, test_acc, svm, train_pred, test_pred = train_fisher_vector_classifier(
                    bovw_model, train_features, train_labels, test_features, test_labels
                )
            
            # Plot confusion matrices
            method_key = f"{layer_name}_{method}"
            plot_confusion_matrix(train_labels, train_pred, class_names,
                                f"bovw_{method_key}_train_confusion_matrix.png",
                                f"BoVW ({layer_name}, {method}) Train CM")
            plot_confusion_matrix(test_labels, test_pred, class_names,
                                f"bovw_{method_key}_test_confusion_matrix.png",
                                f"BoVW ({layer_name}, {method}) Test CM")
            
            bovw_results[method_key] = {
                'train_acc': train_acc,
                'test_acc': test_acc,
                'svm': svm,
                'codebook_size': codebook_size,
                'method': method
            }
            
            print(f"\nBoVW Results ({layer_name} layer, {method}):")
            print(f"  Codebook Size: {codebook_size}")
            print(f"  Train Accuracy: {train_acc:.4f}")
            print(f"  Test Accuracy: {test_acc:.4f}")
            
            # Save results
            with open(f'bovw_{method_key}_results.pkl', 'wb') as f:
                pickle.dump({
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'codebook_size': codebook_size,
                    'method': method
                }, f)
    
    # Comparison
    print("\n" + "="*60)
    print("PART 3: Final Comparison")
    print("="*60 + "\n")
    
    print(f"{'Method':<35} {'Train Acc':<12} {'Test Acc':<12}")
    print("-" * 71)
    print(f"{'End-to-End (Mean Agg)':<35} {end2end_results['train_acc']:.4f}       {end2end_results['best_test_acc']:.4f}")
    print(f"{'BoVW (Hidden, K-means)':<35} {bovw_results['hidden_kmeans']['train_acc']:.4f}       {bovw_results['hidden_kmeans']['test_acc']:.4f}")
    print(f"{'BoVW (Hidden, Fisher Vector)':<35} {bovw_results['hidden_fisher']['train_acc']:.4f}       {bovw_results['hidden_fisher']['test_acc']:.4f}")
    print(f"{'BoVW (Output, K-means)':<35} {bovw_results['output_kmeans']['train_acc']:.4f}       {bovw_results['output_kmeans']['test_acc']:.4f}")
    print(f"{'BoVW (Output, Fisher Vector)':<35} {bovw_results['output_fisher']['train_acc']:.4f}       {bovw_results['output_fisher']['test_acc']:.4f}")
    
    # Find best method
    all_methods = {
        'End-to-End': end2end_results['best_test_acc'],
        'BoVW (Hidden, K-means)': bovw_results['hidden_kmeans']['test_acc'],
        'BoVW (Hidden, Fisher)': bovw_results['hidden_fisher']['test_acc'],
        'BoVW (Output, K-means)': bovw_results['output_kmeans']['test_acc'],
        'BoVW (Output, Fisher)': bovw_results['output_fisher']['test_acc']
    }
    best_method = max(all_methods, key=all_methods.get)
    best_acc = all_methods[best_method]
    
    print("\n" + "="*71)
    print(f"🏆 BEST METHOD: {best_method}")
    print(f"   Test Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print("="*71)
    
    # Create comparison plot
    plot_comparison(end2end_results, bovw_results, save_path="final_comparison.png")
    
    # Save all results
    final_results = {
        'end2end': end2end_results,
        'bovw': bovw_results,
        'config': {
            'patch_size': PATCH_SIZE,
            'codebook_size': args.codebook_size,
            'hidden_dim': 300,
            'num_classes': 11
        }
    }
    
    with open('final_results.pkl', 'wb') as f:
        pickle.dump(final_results, f)
