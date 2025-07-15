"""
Module: first_classifier_predictor.py

Description:
This module implements the inference logic for the first-stage classifier in the Legal 
Hallucination Detector pipeline. It determines whether Hebrew legal text paragraphs 
require further verification by identifying potentially problematic content.

Key Functionality:
1. Model Loading and Inference
   - Loads a fine-tuned Aleph-BERT model specialized for Hebrew legal text
   - Provides efficient tokenization and prediction capabilities
   - Returns both classification result and confidence score

2. Interactive Interface
   - Includes a command-line interface for real-time text evaluation
   - Supports batch processing of predefined examples
   - Displays confidence percentages for user-friendly interpretation

3. Pipeline Integration
   - Acts as the initial filtering step in the hallucination detection workflow
   - Identifies text that requires further analysis with search queries
   - Reduces processing overhead by filtering out content that doesn't need verification

This module represents the entry point of the hallucination detection pipeline, making
the initial determination of which legal paragraphs warrant deeper investigation for
potential factual inaccuracies.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Path to your fine-tuned model
MODEL_PATH = "aleph_bert_finetuned"

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the fine-tuned model and tokenizer
print("Loading fine-tuned Aleph-BERT model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()  # Set the model to evaluation mode


def predict_needs_check(text):
    """
    Predict whether the given Hebrew text needs checking.

    Args:
        text (str): Hebrew text paragraph

    Returns:
        tuple: (prediction (0 or 1), confidence score)
    """
    # Tokenize the input text
    encoding = tokenizer(
        text,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Move the encoded inputs to the device
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Get prediction class and probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][prediction].item()

    return prediction, confidence


def main():
    print("Hebrew Text Classification - Needs Check Predictor")
    print("=" * 50)
    print("Enter Hebrew text to classify (or type 'exit' to quit):")

    while True:
        # Get text input from user
        text = input("\nEnter text: ")

        if text.lower() == 'exit':
            break

        if not text.strip():
            print("Please enter some text to classify.")
            continue

        # Make prediction
        prediction, confidence = predict_needs_check(text)

        # Display result
        result = "NEEDS CHECK" if prediction == 1 else "DOES NOT NEED CHECK"
        print(f"\nPrediction: {result}")
        print(f"Confidence: {confidence:.4f} ({confidence * 100:.2f}%)")

    print("\nExiting...")


# Example texts to test with (you can replace these with your own examples)
EXAMPLE_TEXTS = [
    # Add your example texts here
    # "יש לבדוק האם המידע הזה נכון או לא.",
    # "זוהי עובדה ידועה שהשמש זורחת במזרח."
]

if __name__ == "__main__":
    # You can either run the interactive mode:
    main()

    # Or test with predefined examples:
    """
    print("\nTesting with example texts:")
    print("=" * 50)
    for i, text in enumerate(EXAMPLE_TEXTS):
        prediction, confidence = predict_needs_check(text)
        result = "NEEDS CHECK" if prediction == 1 else "DOES NOT NEED CHECK"
        print(f"\nExample {i+1}:")
        print(f"Text: {text}")
        print(f"Prediction: {result}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    """
