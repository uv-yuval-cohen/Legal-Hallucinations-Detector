import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import random
from datetime import datetime
from tqdm import tqdm

# Set random seeds for reproducibility
SEED = 48
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)

# Configuration
CONFIG = {
    'data_file': 'hebrew_hallucination_embeddings.npz',
    'batch_size': 4,  # Small batch size for small dataset
    'learning_rate': 0.001,
    'weight_decay': 1e-4,  # L2 regularization
    'max_epochs': 200,
    'patience': 10,  # Early stopping patience
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15
}


class HallucinationDataset(Dataset):
    """Custom dataset for Hebrew hallucination detection."""

    def __init__(self, original_embeddings, search_embeddings, labels):
        # Ensure index correspondence is maintained
        assert len(original_embeddings) == len(search_embeddings) == len(labels)

        # Concatenate embeddings: [original_emb, search_emb]
        self.features = np.concatenate([original_embeddings, search_embeddings], axis=1)
        self.labels = labels

        # Convert to tensors and ensure proper shapes
        self.features = torch.FloatTensor(self.features)
        self.labels = torch.FloatTensor(labels).view(-1)  # Ensure 1D shape

        print(f"Dataset created - Features shape: {self.features.shape}, Labels shape: {self.labels.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class HallucinationClassifier(nn.Module):
    """Simple dense network with heavy regularization for small dataset."""

    def __init__(self, input_dim=2048, dropout_rates=[0.5, 0.3]):
        super(HallucinationClassifier, self).__init__()

        self.network = nn.Sequential(
            # First layer: 2048 -> 256
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rates[0]),

            # Second layer: 256 -> 64
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rates[1]),

            # Output layer: 64 -> 1
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x).squeeze(-1)  # Only squeeze the last dimension


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict().copy()
        else:
            self.counter += 1

        return self.counter >= self.patience


def load_data(file_path):
    """Load and validate NPZ data."""
    print(f"Loading data from {file_path}...")

    try:
        data = np.load(file_path)
        original_embeddings = data['original_embeddings']
        search_embeddings = data['search_embeddings']
        labels = data['labels']
        indices = data['indices']

        print(f"Data loaded successfully!")
        print(f"Original embeddings shape: {original_embeddings.shape}")
        print(f"Search embeddings shape: {search_embeddings.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Indices shape: {indices.shape}")

        # Validate data
        assert original_embeddings.shape[0] == search_embeddings.shape[0] == len(labels)
        assert original_embeddings.shape[1] == search_embeddings.shape[1] == 1024

        # Check label balance
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Label distribution: {dict(zip(unique, counts))}")

        return original_embeddings, search_embeddings, labels, indices

    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def create_data_splits(original_embeddings, search_embeddings, labels, indices, config):
    """Create stratified train/val/test splits."""
    print("Creating data splits...")

    # First split: train vs (val + test)
    train_orig, temp_orig, train_search, temp_search, train_labels, temp_labels, train_idx, temp_idx = train_test_split(
        original_embeddings, search_embeddings, labels, indices,
        test_size=(config['val_ratio'] + config['test_ratio']),
        random_state=SEED,
        stratify=labels
    )

    # Second split: val vs test
    val_size = config['val_ratio'] / (config['val_ratio'] + config['test_ratio'])
    val_orig, test_orig, val_search, test_search, val_labels, test_labels, val_idx, test_idx = train_test_split(
        temp_orig, temp_search, temp_labels, temp_idx,
        test_size=(1 - val_size),
        random_state=SEED,
        stratify=temp_labels
    )

    print(f"Train set: {len(train_labels)} samples")
    print(f"Validation set: {len(val_labels)} samples")
    print(f"Test set: {len(test_labels)} samples")

    # Create datasets
    train_dataset = HallucinationDataset(train_orig, train_search, train_labels)
    val_dataset = HallucinationDataset(val_orig, val_search, val_labels)
    test_dataset = HallucinationDataset(test_orig, test_search, test_labels)

    return train_dataset, val_dataset, test_dataset, (train_idx, val_idx, test_idx)


def train_model(model, train_loader, val_loader, config, device):
    """Train the model with early stopping."""
    print("Starting training...")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=config['learning_rate'],
                           weight_decay=config['weight_decay'])

    early_stopping = EarlyStopping(patience=config['patience'])

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    best_epoch = 0

    # Progress bar for epochs only
    epoch_pbar = tqdm(range(config['max_epochs']), desc="Training Progress")

    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)

            # Debug shapes for first batch
            if epoch == 0 and batch_idx == 0:
                print(f"First batch - Features shape: {features.shape}, Labels shape: {labels.shape}")

            optimizer.zero_grad()
            outputs = model(features)

            # Debug output shape for first batch
            if epoch == 0 and batch_idx == 0:
                print(f"First batch - Outputs shape: {outputs.shape}")

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate averages
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Update epoch progress bar with epoch results
        epoch_pbar.set_postfix({
            'Train Loss': f'{train_loss:.4f}',
            'Val Loss': f'{val_loss:.4f}',
            'Train Acc': f'{train_acc:.4f}',
            'Val Acc': f'{val_acc:.4f}'
        })

        # Print detailed epoch results
        print(f'Epoch {epoch + 1:3d}/{config["max_epochs"]} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

        # Early stopping check
        if early_stopping(val_loss, model):
            best_epoch = epoch + 1 - early_stopping.counter + 1  # Calculate best epoch
            print(f'\nEarly stopping at epoch {epoch + 1}')
            print(f'Loading best model from epoch {best_epoch} (Val Loss: {early_stopping.best_loss:.4f})')
            model.load_state_dict(early_stopping.best_model_state)
            break

        # Update best epoch if this is the current best
        if val_loss == early_stopping.best_loss:
            best_epoch = epoch + 1

    epoch_pbar.close()

    # If we completed all epochs without early stopping
    if best_epoch == 0:
        best_epoch = len(history['val_loss'])
        min_val_loss_idx = np.argmin(history['val_loss'])
        best_epoch = min_val_loss_idx + 1
        print(
            f'\nTraining completed. Best model from epoch {best_epoch} (Val Loss: {history["val_loss"][min_val_loss_idx]:.4f})')

    return model, history


def evaluate_model(model, test_loader, device, test_indices=None):
    """Evaluate model on test set with index tracking."""
    print("\nEvaluating model on test set...")

    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)

            probabilities = outputs.cpu().numpy()
            predictions = (outputs > 0.5).float().cpu().numpy()

            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    roc_auc = roc_auc_score(all_labels, all_probabilities)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions,
                                target_names=['Non-Hallucination', 'Hallucination']))

    # NEW: Track correct vs incorrect predictions by index
    if test_indices is not None:
        correct_mask = (all_predictions == all_labels)

        # Get the original indices of correct and incorrect predictions
        correct_original_indices = test_indices[correct_mask]
        incorrect_original_indices = test_indices[~correct_mask]

        print(f"\n{'=' * 50}")
        print("INDEX TRACKING RESULTS")
        print(f"{'=' * 50}")
        print(f"Total test samples: {len(all_predictions)}")
        print(f"Correct predictions: {len(correct_original_indices)}")
        print(f"Incorrect predictions: {len(incorrect_original_indices)}")

        print(f"\nOriginal indices of CORRECT predictions:")
        print(f"{sorted(correct_original_indices.tolist())}")

        print(f"\nOriginal indices of INCORRECT predictions:")
        print(f"{sorted(incorrect_original_indices.tolist())}")

        # Analyze the incorrect predictions in detail
        print(f"\nDETAILS OF INCORRECT PREDICTIONS:")
        for i, orig_idx in enumerate(incorrect_original_indices):
            # Find position in test set
            pos_in_test = np.where(test_indices == orig_idx)[0][0]
            true_label = all_labels[pos_in_test]
            pred_label = all_predictions[pos_in_test]
            probability = all_probabilities[pos_in_test]
            confidence = abs(probability - 0.5)  # Distance from decision boundary

            error_type = "False Positive" if true_label == 0 else "False Negative"
            print(f"  Index {orig_idx}: {error_type} - True: {int(true_label)}, "
                  f"Pred: {int(pred_label)}, Prob: {probability:.3f}, Confidence: {confidence:.3f}")

        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            'timestamp': timestamp,
            'accuracy': float(accuracy),
            'roc_auc': float(roc_auc),
            'correct_indices': correct_original_indices.tolist(),
            'incorrect_indices': incorrect_original_indices.tolist(),
            'incorrect_details': [
                {
                    'original_index': int(orig_idx),
                    'true_label': int(all_labels[np.where(test_indices == orig_idx)[0][0]]),
                    'predicted_label': int(all_predictions[np.where(test_indices == orig_idx)[0][0]]),
                    'probability': float(all_probabilities[np.where(test_indices == orig_idx)[0][0]]),
                    'error_type': "False Positive" if all_labels[np.where(test_indices == orig_idx)[0][
                        0]] == 0 else "False Negative"
                }
                for orig_idx in incorrect_original_indices
            ]
        }

        with open(f'index_tracking_results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: index_tracking_results_{timestamp}.json")

        return all_predictions, all_probabilities, all_labels, {
            'correct_indices': correct_original_indices,
            'incorrect_indices': incorrect_original_indices
        }

    return all_predictions, all_probabilities, all_labels


def plot_training_curves(history):
    """Plot training and validation curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss curves
    ax1.plot(history['train_loss'], label='Training Loss', marker='o', markersize=3)
    ax1.plot(history['val_loss'], label='Validation Loss', marker='s', markersize=3)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2.plot(history['train_acc'], label='Training Accuracy', marker='o', markersize=3)
    ax2.plot(history['val_acc'], label='Validation Accuracy', marker='s', markersize=3)
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_evaluation_results(predictions, probabilities, labels):
    """Plot confusion matrix and ROC curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Confusion Matrix
    cm = confusion_matrix(labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Non-Hallucination', 'Hallucination'],
                yticklabels=['Non-Hallucination', 'Hallucination'])
    ax1.set_title('Confusion Matrix')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')

    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = roc_auc_score(labels, probabilities)

    ax2.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_model_and_results(model, history, config, predictions, probabilities, labels, indices):
    """Save trained model and results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model
    model_path = f'hebrew_hallucination_classifier_{timestamp}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'timestamp': timestamp
    }, model_path)

    # Save results
    results = {
        'config': config,
        'timestamp': timestamp,
        'final_accuracy': accuracy_score(labels, predictions),
        'final_roc_auc': roc_auc_score(labels, probabilities),
        'training_history': history,
        'test_predictions': predictions.tolist(),
        'test_probabilities': probabilities.tolist(),
        'test_labels': labels.tolist()
    }

    results_path = f'training_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Model saved to: {model_path}")
    print(f"Results saved to: {results_path}")


def main():
    """Main training pipeline."""
    print("Hebrew Hallucination Detection Classifier")
    print("=" * 50)

    # Set device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS device (Mac M4 Pro GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA device")
    else:
        device = torch.device('cpu')
        print("Using CPU device")

    # Load data
    original_emb, search_emb, labels, indices = load_data(CONFIG['data_file'])

    # Create data splits
    train_dataset, val_dataset, test_dataset, split_indices = create_data_splits(
        original_emb, search_emb, labels, indices, CONFIG
    )
    train_idx, val_idx, test_idx = split_indices  # NEW: Extract test indices

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                              shuffle=True, drop_last=True)  # Drop incomplete batches in training
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                            shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'],
                             shuffle=False, drop_last=False)

    # Initialize model
    model = HallucinationClassifier().to(device)
    print(f"\nModel architecture:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Train model
    model, history = train_model(model, train_loader, val_loader, CONFIG, device)

    # Evaluate model with index tracking
    eval_results = evaluate_model(model, test_loader, device, test_indices=test_idx)

    if len(eval_results) == 4:  # With index tracking
        predictions, probabilities, test_labels, index_results = eval_results
        correct_indices = index_results['correct_indices']
        incorrect_indices = index_results['incorrect_indices']

        print(f"\nModel got these samples WRONG: {sorted(incorrect_indices.tolist())}")
        print(f"Model got these samples RIGHT: {sorted(correct_indices.tolist())}")
    else:  # Without index tracking
        predictions, probabilities, test_labels = eval_results

    # Plot results
    plot_training_curves(history)
    plot_evaluation_results(predictions, probabilities, test_labels)

    # Save model and results
    save_model_and_results(model, history, CONFIG, predictions, probabilities,
                           test_labels, split_indices)

    print("\nTraining completed successfully!")
    print(f"Final Test Accuracy: {accuracy_score(test_labels, predictions):.4f}")
    print(f"Final Test ROC-AUC: {roc_auc_score(test_labels, probabilities):.4f}")


if __name__ == "__main__":
    main()