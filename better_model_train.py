import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import random
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ~~~~~~~~~~~~ TRAINING FIRST CLASSIFIER ~~~~~~~~~~~~


# Set random seed for reproducibility
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


set_seed(42)

# Define parameters
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 6
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01  # For regularization to prevent overfitting
EARLY_STOPPING_PATIENCE = 3  # Number of epochs to wait for improvement before stopping
MODEL_NAME = "onlplab/alephbert-base"  # Aleph-BERT model for Hebrew
OUTPUT_DIR = "aleph_bert_finetuned"

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load and prepare data with train/val/test split
def load_data(file_path):
    print("Loading data...")
    df = pd.read_csv(file_path)

    # Display basic info
    print(f"Dataset shape: {df.shape}")
    print(f"need_check distribution: {df['need_check'].value_counts().to_dict()}")

    # Convert labels to integers if needed
    if df['need_check'].dtype != int:
        df['need_check'] = df['need_check'].astype(int)

    # First split data into train and temp sets (80% / 20%)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['need_check']
    )

    # Then split temp into validation and test sets (10% / 10% of original data)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df['need_check']
    )

    print(f"Training set: {train_df.shape} ({train_df.shape[0] / df.shape[0] * 100:.1f}%)")
    print(f"Validation set: {val_df.shape} ({val_df.shape[0] / df.shape[0] * 100:.1f}%)")
    print(f"Test set: {test_df.shape} ({test_df.shape[0] / df.shape[0] * 100:.1f}%)")

    # Check class balance in each split
    print("\nClass distribution:")
    print(f"Training: {train_df['need_check'].value_counts().to_dict()}")
    print(f"Validation: {val_df['need_check'].value_counts().to_dict()}")
    print(f"Test: {test_df['need_check'].value_counts().to_dict()}")

    return train_df, val_df, test_df


# Create a custom dataset
class HebrewTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# Evaluate model performance
def evaluate_model(model, data_loader):
    model.eval()

    predictions = []
    actual_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs.logits, dim=1)

            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())

    # Calculate metrics
    accuracy = accuracy_score(actual_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual_labels, predictions, average='binary', zero_division=0
    )

    # Generate confusion matrix
    cm = confusion_matrix(actual_labels, predictions)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': predictions,
        'actual_labels': actual_labels
    }


# Plot training history
def plot_training_history(train_losses, val_metrics, output_dir):
    # Create directory for plots if it doesn't exist
    plots_dir = os.path.join(output_dir, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'training_loss.png'))
    plt.close()

    # Plot validation metrics
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(val_metrics['accuracy']) + 1)
    plt.plot(epochs, val_metrics['accuracy'], 'b-', label='Accuracy')
    plt.plot(epochs, val_metrics['precision'], 'g-', label='Precision')
    plt.plot(epochs, val_metrics['recall'], 'r-', label='Recall')
    plt.plot(epochs, val_metrics['f1'], 'y-', label='F1 Score')
    plt.title('Validation Metrics Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'validation_metrics.png'))
    plt.close()


# Plot confusion matrix
def plot_confusion_matrix(cm, output_dir, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, "plots", f"{title.lower().replace(' ', '_')}.png"))
    plt.close()


# Main training function
def train():
    # Load data with three-way split
    train_df, val_df, test_df = load_data('annotated_paragraphs.csv')

    # Load tokenizer and model with fresh initialization for classification layer
    print("Loading Aleph-BERT tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Initialize model with explicit reset of classifier weights
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,  # Binary classification
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True  # Ignore previous classifier head size if exists
    )

    # Explicitly reinitialize the classifier layer to ensure fresh start
    print("Reinitializing classification head with random weights...")
    model.classifier.weight.data.normal_(mean=0.0, std=0.02)
    model.classifier.bias.data.zero_()

    # Print model architecture to confirm
    print(f"Model architecture:\n{model.__class__.__name__}")
    print(f"Classification head initialized with random weights")

    model.to(device)

    # Create datasets
    train_dataset = HebrewTextDataset(
        texts=train_df['paragraph'].values,
        labels=train_df['need_check'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    val_dataset = HebrewTextDataset(
        texts=val_df['paragraph'].values,
        labels=val_df['need_check'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    test_dataset = HebrewTextDataset(
        texts=test_df['paragraph'].values,
        labels=test_df['need_check'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    # Create data loaders
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=BATCH_SIZE
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE
    )

    # Prepare optimizer and scheduler
    # Add weight decay to certain parameters (but not biases and layer norms) for regularization
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': WEIGHT_DECAY
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),  # 10% of total steps for warmup
        num_training_steps=total_steps
    )

    # Training loop
    print(f"Starting training for {EPOCHS} epochs...")
    best_f1 = 0
    no_improvement_count = 0

    train_losses = []
    val_metrics_history = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        # Training
        model.train()
        total_loss = 0

        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            model.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            if (step + 1) % 100 == 0 or (step + 1) == len(train_dataloader):
                print(f"Batch {step + 1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Validation
        print("Evaluating on validation set...")
        metrics = evaluate_model(model, val_dataloader)

        # Store metrics history
        for key in val_metrics_history.keys():
            val_metrics_history[key].append(metrics[key])

        print(
            f"Validation Metrics - Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")

        # Save best model and check for early stopping
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            no_improvement_count = 0
            print(f"New best F1 score: {best_f1:.4f} - Saving model")

            # Create output directory
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            # Save model and tokenizer
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
        else:
            no_improvement_count += 1
            print(f"No improvement in F1 score for {no_improvement_count} epochs")

            if no_improvement_count >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # Create plots directory
    plots_dir = os.path.join(OUTPUT_DIR, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Plot training history
    plot_training_history(train_losses, val_metrics_history, OUTPUT_DIR)

    # Final evaluation on test set
    print("\nFinal evaluation on test set...")

    # Load the best model for final evaluation
    print("Loading best model for final evaluation...")
    best_model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR)
    best_model.to(device)

    # Verify we're using a different model instance than the training one
    print(f"Using best saved model from {OUTPUT_DIR} for evaluation")

    test_metrics = evaluate_model(best_model, test_dataloader)
    print(
        f"Test Metrics - Accuracy: {test_metrics['accuracy']:.4f}, Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")

    # Plot test confusion matrix
    plot_confusion_matrix(test_metrics['confusion_matrix'], OUTPUT_DIR, 'Test Confusion Matrix')

    # Save test metrics to file
    with open(os.path.join(OUTPUT_DIR, "test_results.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"Test Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"Test Recall: {test_metrics['recall']:.4f}\n")
        f.write(f"Test F1 Score: {test_metrics['f1']:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(test_metrics['confusion_matrix']))

    print(f"\nTraining completed! Best validation F1 score: {best_f1:.4f}")
    print(f"Model saved to {OUTPUT_DIR}")
    print(f"Test results saved to {os.path.join(OUTPUT_DIR, 'test_results.txt')}")

    return best_model, tokenizer


# Run the training process
if __name__ == "__main__":
    train()