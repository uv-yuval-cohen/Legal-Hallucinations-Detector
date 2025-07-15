"""
Module: hallucination_classifier_kfold_trainer.py

Description:
This module implements the final stage hallucination classifier in the Legal Hallucination 
Detector pipeline, using k-fold cross-validation for robust model evaluation. It processes 
embedding vectors from legal texts and their search results to make binary hallucination 
determinations.

Key Functionality:
1. Enhanced Feature Engineering
   - Creates 4096-dimensional feature vectors by combining original and search embeddings
   - Implements element-wise product and absolute difference operations for semantic comparison

2. Model Architecture
   - Neural network classifier with regularization layers optimized for small datasets
   - Implements early stopping to prevent overfitting
   - Xavier uniform weight initialization for stable training

3. K-Fold Cross-Validation
   - Stratified k-fold training to handle class imbalance
   - Comprehensive metrics tracking across folds
   - Detailed per-sample error analysis with index tracking

4. Evaluation and Analysis
   - Performance visualization with fold-by-fold comparisons
   - Final model evaluation on held-out test set
   - Error analysis identifying consistently problematic examples

This module represents the final decision-making component in the hallucination detection 
pipeline, producing a classifier that achieves strong performance in identifying factual 
inaccuracies in legal text.
"""

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
import os

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)

# Configuration
CONFIG = {
    'data_file': 'hebrew_hallucination_embeddings.npz',
    'batch_size': 4,  # Based on your good results
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'max_epochs': 200,
    'patience': 15,
    'k_folds': 5,
    'test_ratio': 0.15,  # Hold out test set separate from CV
}


class HallucinationDataset(Dataset):
    """Custom dataset for Hebrew hallucination detection."""

    def __init__(self, original_embeddings, search_embeddings, labels):
        # Ensure index correspondence is maintained
        assert len(original_embeddings) == len(search_embeddings) == len(labels)

        # Enhanced feature combination: [original, search, original*search, |original-search|]
        element_product = original_embeddings * search_embeddings
        element_diff = np.abs(original_embeddings - search_embeddings)

        self.features = np.concatenate([
            original_embeddings,  # 1024 dims
            search_embeddings,  # 1024 dims
            element_product,  # 1024 dims
            element_diff  # 1024 dims
        ], axis=1)  # Total: 4096 dims

        self.labels = labels

        print(f"Enhanced features shape: {self.features.shape} (4096 dims expected)")

        # Convert to tensors and ensure proper shapes
        self.features = torch.FloatTensor(self.features)
        self.labels = torch.FloatTensor(labels).view(-1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class HallucinationClassifier(nn.Module):
    """Simple dense network with heavy regularization for small dataset."""

    def __init__(self, input_dim=4096, dropout_rates=[0.5, 0.3]):  # Updated input_dim
        super(HallucinationClassifier, self).__init__()

        self.network = nn.Sequential(
            # First layer: 4096 -> 256 (adjusted for enhanced features)
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
        return self.network(x).squeeze(-1)


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


def create_results_folder():
    """Create organized folder structure for k-fold results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"kfold_enhanced_features_{timestamp}")

    # Create subdirectories
    (results_dir / "fold_models").mkdir(parents=True)
    (results_dir / "fold_plots").mkdir(parents=True)
    (results_dir / "fold_results").mkdir(parents=True)

    return results_dir


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


def train_single_fold(model, train_loader, val_loader, config, device, fold_num):
    """Train model for a single fold."""
    print(f"\n--- Training Fold {fold_num} ---")

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

    # Progress bar for epochs
    epoch_pbar = tqdm(range(config['max_epochs']), desc=f"Fold {fold_num} Training")

    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
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

        # Update progress bar
        epoch_pbar.set_postfix({
            'Train Loss': f'{train_loss:.4f}',
            'Val Loss': f'{val_loss:.4f}',
            'Train Acc': f'{train_acc:.4f}',
            'Val Acc': f'{val_acc:.4f}'
        })

        # Early stopping check
        if early_stopping(val_loss, model):
            best_epoch = epoch + 1 - early_stopping.counter + 1
            print(f'Early stopping at epoch {epoch + 1}')
            print(f'Loading best model from epoch {best_epoch} (Val Loss: {early_stopping.best_loss:.4f})')
            model.load_state_dict(early_stopping.best_model_state)
            break

        if val_loss == early_stopping.best_loss:
            best_epoch = epoch + 1

    epoch_pbar.close()

    if best_epoch == 0:
        best_epoch = len(history['val_loss'])
        min_val_loss_idx = np.argmin(history['val_loss'])
        best_epoch = min_val_loss_idx + 1

    return model, history, best_epoch


def evaluate_fold(model, val_loader, device, val_indices=None, fold_num=None):
    """Evaluate model on validation fold with index tracking."""
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for features, labels in val_loader:
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

    # NEW: Track correct vs incorrect predictions by index for this fold
    index_results = None
    if val_indices is not None and fold_num is not None:
        correct_mask = (all_predictions == all_labels)

        # Get the original indices of correct and incorrect predictions for this fold
        correct_original_indices = val_indices[correct_mask]
        incorrect_original_indices = val_indices[~correct_mask]

        print(f"\n--- Fold {fold_num} Index Tracking ---")
        print(f"Validation samples: {len(all_predictions)}")
        print(f"Correct predictions: {len(correct_original_indices)}")
        print(f"Incorrect predictions: {len(incorrect_original_indices)}")

        if len(incorrect_original_indices) > 0:
            print(f"Incorrect indices in fold {fold_num}: {sorted(incorrect_original_indices.tolist())}")

            # Analyze incorrect predictions
            print(f"Details of incorrect predictions in fold {fold_num}:")
            for i, orig_idx in enumerate(incorrect_original_indices):
                pos_in_val = np.where(val_indices == orig_idx)[0][0]
                true_label = all_labels[pos_in_val]
                pred_label = all_predictions[pos_in_val]
                probability = all_probabilities[pos_in_val]
                confidence = abs(probability - 0.5)

                error_type = "False Positive" if true_label == 0 else "False Negative"
                print(f"  Index {orig_idx}: {error_type} - True: {int(true_label)}, "
                      f"Pred: {int(pred_label)}, Prob: {probability:.3f}, Confidence: {confidence:.3f}")

        index_results = {
            'correct_indices': correct_original_indices.tolist(),
            'incorrect_indices': incorrect_original_indices.tolist(),
            'incorrect_details': [
                {
                    'original_index': int(orig_idx),
                    'true_label': int(all_labels[np.where(val_indices == orig_idx)[0][0]]),
                    'predicted_label': int(all_predictions[np.where(val_indices == orig_idx)[0][0]]),
                    'probability': float(all_probabilities[np.where(val_indices == orig_idx)[0][0]]),
                    'error_type': "False Positive" if all_labels[np.where(val_indices == orig_idx)[0][
                        0]] == 0 else "False Negative"
                }
                for orig_idx in incorrect_original_indices
            ]
        }

    return accuracy, roc_auc, all_predictions, all_probabilities, all_labels, index_results


def plot_fold_results(history, fold_num, results_dir):
    """Plot training curves for a single fold."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss curves
    ax1.plot(history['train_loss'], label='Training Loss', marker='o', markersize=3)
    ax1.plot(history['val_loss'], label='Validation Loss', marker='s', markersize=3)
    ax1.set_title(f'Fold {fold_num} - Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2.plot(history['train_acc'], label='Training Accuracy', marker='o', markersize=3)
    ax2.plot(history['val_acc'], label='Validation Accuracy', marker='s', markersize=3)
    ax2.set_title(f'Fold {fold_num} - Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "fold_plots" / f'fold_{fold_num}_training_curves.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_cv_summary(fold_results, results_dir):
    """Plot summary of all folds performance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Extract metrics
    accuracies = [result['accuracy'] for result in fold_results]
    roc_aucs = [result['roc_auc'] for result in fold_results]
    folds = list(range(1, len(fold_results) + 1))

    # Accuracy plot
    ax1.bar(folds, accuracies, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.axhline(y=np.mean(accuracies), color='red', linestyle='--',
                label=f'Mean: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}')
    ax1.set_title('Cross-Validation Accuracy by Fold')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ROC-AUC plot
    ax2.bar(folds, roc_aucs, alpha=0.7, color='lightcoral', edgecolor='darkred')
    ax2.axhline(y=np.mean(roc_aucs), color='blue', linestyle='--',
                label=f'Mean: {np.mean(roc_aucs):.3f} ± {np.std(roc_aucs):.3f}')
    ax2.set_title('Cross-Validation ROC-AUC by Fold')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('ROC-AUC')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / 'cv_summary.png', dpi=300, bbox_inches='tight')
    plt.show()


def k_fold_cross_validation(original_emb, search_emb, labels, indices, config, device, results_dir):
    """Perform k-fold cross validation with index tracking."""
    print(f"\n{'=' * 60}")
    print(f"Starting {config['k_folds']}-Fold Cross Validation with Index Tracking")
    print(f"{'=' * 60}")

    # Hold out test set first
    train_orig, test_orig, train_search, test_search, train_labels, test_labels, train_idx, test_idx = train_test_split(
        original_emb, search_emb, labels, indices,
        test_size=config['test_ratio'],
        random_state=SEED,
        stratify=labels
    )

    print(f"CV set: {len(train_labels)} samples")
    print(f"Final test set: {len(test_labels)} samples (held out)")

    # Initialize k-fold
    skf = StratifiedKFold(n_splits=config['k_folds'], shuffle=True, random_state=SEED)
    fold_results = []

    # Track all incorrect indices across folds
    all_incorrect_indices = set()

    # Perform k-fold CV
    for fold_num, (train_indices, val_indices) in enumerate(skf.split(train_orig, train_labels), 1):
        print(f"\n{'-' * 40}")
        print(f"FOLD {fold_num}/{config['k_folds']}")
        print(f"{'-' * 40}")

        # Create fold datasets
        fold_train_orig = train_orig[train_indices]
        fold_train_search = train_search[train_indices]
        fold_train_labels = train_labels[train_indices]

        fold_val_orig = train_orig[val_indices]
        fold_val_search = train_search[val_indices]
        fold_val_labels = train_labels[val_indices]
        fold_val_indices = train_idx[val_indices]  # NEW: Get original indices for validation set

        print(f"Fold {fold_num} - Train: {len(fold_train_labels)}, Val: {len(fold_val_labels)}")

        # Create datasets and dataloaders
        fold_train_dataset = HallucinationDataset(fold_train_orig, fold_train_search, fold_train_labels)
        fold_val_dataset = HallucinationDataset(fold_val_orig, fold_val_search, fold_val_labels)

        train_loader = DataLoader(fold_train_dataset, batch_size=config['batch_size'],
                                  shuffle=True, drop_last=True)
        val_loader = DataLoader(fold_val_dataset, batch_size=config['batch_size'],
                                shuffle=False, drop_last=False)

        # Initialize and train model
        model = HallucinationClassifier().to(device)
        model, history, best_epoch = train_single_fold(model, train_loader, val_loader,
                                                       config, device, fold_num)

        # Evaluate fold with index tracking
        accuracy, roc_auc, predictions, probabilities, val_labels, index_results = evaluate_fold(
            model, val_loader, device, val_indices=fold_val_indices, fold_num=fold_num
        )

        print(f"Fold {fold_num} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  Best Epoch: {best_epoch}")

        # Store fold results
        fold_result = {
            'fold': fold_num,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'best_epoch': best_epoch,
            'history': history,
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'true_labels': val_labels.tolist(),
            'index_tracking': index_results  # NEW: Add index tracking results
        }
        fold_results.append(fold_result)

        # Collect incorrect indices across all folds
        if index_results:
            all_incorrect_indices.update(index_results['incorrect_indices'])

        # Save fold model and results
        torch.save({
            'model_state_dict': model.state_dict(),
            'fold': fold_num,
            'config': config,
            'best_epoch': best_epoch,
            'accuracy': accuracy,
            'roc_auc': roc_auc
        }, results_dir / "fold_models" / f'fold_{fold_num}_model.pth')

        # Save fold results
        with open(results_dir / "fold_results" / f'fold_{fold_num}_results.json', 'w') as f:
            json.dump(fold_result, f, indent=2)

        # Plot fold training curves
        plot_fold_results(history, fold_num, results_dir)

    # NEW: Summary of indices that were incorrect across folds
    print(f"\n{'=' * 60}")
    print("CROSS-VALIDATION INDEX SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total unique indices that were incorrect across all folds: {len(all_incorrect_indices)}")
    print(f"Consistently problematic indices: {sorted(list(all_incorrect_indices))}")

    # Count how many times each index was incorrect
    index_error_counts = {}
    for fold_result in fold_results:
        if fold_result['index_tracking']:
            for idx in fold_result['index_tracking']['incorrect_indices']:
                index_error_counts[idx] = index_error_counts.get(idx, 0) + 1

    # Find indices that were wrong multiple times
    frequent_errors = {idx: count for idx, count in index_error_counts.items() if count > 1}
    if frequent_errors:
        print(f"\nIndices incorrect in multiple folds:")
        for idx, count in sorted(frequent_errors.items(), key=lambda x: x[1], reverse=True):
            print(f"  Index {idx}: incorrect in {count} fold(s)")

    return fold_results, (train_orig, train_search, train_labels, train_idx), (
    test_orig, test_search, test_labels, test_idx)


def final_evaluation(cv_data, test_data, config, device, results_dir):
    """Train final model on all CV data and evaluate on test set with index tracking."""
    print(f"\n{'=' * 60}")
    print("Final Model Training on All CV Data")
    print(f"{'=' * 60}")

    train_orig, train_search, train_labels, train_idx = cv_data
    test_orig, test_search, test_labels, test_idx = test_data

    # Create datasets
    train_dataset = HallucinationDataset(train_orig, train_search, train_labels)
    test_dataset = HallucinationDataset(test_orig, test_search, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                             shuffle=False, drop_last=False)

    # Train final model (without validation - use all CV data)
    model = HallucinationClassifier().to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=config['learning_rate'],
                           weight_decay=config['weight_decay'])

    print("Training final model...")
    for epoch in tqdm(range(50), desc="Final Training"):  # Shorter training
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate on test set with index tracking
    print("\nEvaluating final model on held-out test set...")
    accuracy, roc_auc, predictions, probabilities, true_labels, index_results = evaluate_fold(
        model, test_loader, device, val_indices=test_idx, fold_num="Final"
    )

    print(f"Final Test Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions,
                                target_names=['Non-Hallucination', 'Hallucination']))

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_test_accuracy': accuracy,
        'final_test_roc_auc': roc_auc,
        'final_test_index_tracking': index_results  # NEW: Save final test index tracking
    }, results_dir / 'final_model.pth')

    # Save final test index tracking results
    if index_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(results_dir / f'final_test_index_tracking_{timestamp}.json', 'w') as f:
            json.dump({
                'final_test_accuracy': float(accuracy),
                'final_test_roc_auc': float(roc_auc),
                'index_tracking': index_results,
                'timestamp': timestamp
            }, f, indent=2)

    return accuracy, roc_auc, index_results


def main():
    """Main k-fold cross validation pipeline."""
    print("Hebrew Hallucination Detection Classifier - K-Fold CV (Enhanced Features + Index Tracking)")
    print("=" * 80)
    print("Using enhanced feature combination: [original, search, original*search, |original-search|]")
    print("Input dimensions: 4096 (4 x 1024)")
    print("With comprehensive index tracking for error analysis")
    print("=" * 80)

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

    # Create results folder
    results_dir = create_results_folder()
    print(f"Results will be saved to: {results_dir}")

    # Load data
    original_emb, search_emb, labels, indices = load_data(CONFIG['data_file'])

    # Perform k-fold cross validation
    fold_results, cv_data, test_data = k_fold_cross_validation(
        original_emb, search_emb, labels, indices, CONFIG, device, results_dir
    )

    # Calculate and display CV summary
    print(f"\n{'=' * 60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'=' * 60}")

    accuracies = [result['accuracy'] for result in fold_results]
    roc_aucs = [result['roc_auc'] for result in fold_results]

    print(f"Accuracy:  {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"ROC-AUC:   {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")
    print(f"Min Accuracy: {np.min(accuracies):.4f}")
    print(f"Max Accuracy: {np.max(accuracies):.4f}")

    # Plot CV summary
    plot_cv_summary(fold_results, results_dir)

    # Final evaluation on test set
    final_test_acc, final_test_roc, final_index_results = final_evaluation(cv_data, test_data, CONFIG, device,
                                                                           results_dir)

    # Save complete results
    complete_results = {
        'config': CONFIG,
        'cv_results': {
            'mean_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies)),
            'mean_roc_auc': float(np.mean(roc_aucs)),
            'std_roc_auc': float(np.std(roc_aucs)),
            'individual_folds': fold_results
        },
        'final_test_results': {
            'accuracy': final_test_acc,
            'roc_auc': final_test_roc,
            'index_tracking': final_index_results  # NEW: Include final test index tracking
        },
        'timestamp': datetime.now().isoformat()
    }

    with open(results_dir / 'complete_results.json', 'w') as f:
        json.dump(complete_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"Cross-Validation Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Cross-Validation ROC-AUC:  {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")
    print(f"Final Test Accuracy:       {final_test_acc:.4f}")
    print(f"Final Test ROC-AUC:        {final_test_roc:.4f}")

    # NEW: Final index tracking summary
    if final_index_results:
        print(f"\nFinal Test Set - Incorrect Indices: {sorted(final_index_results['incorrect_indices'])}")

    print(f"\nAll results saved to: {results_dir}")


if __name__ == "__main__":
    main()
