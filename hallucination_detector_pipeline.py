"""
Module: hallucination_detector.py

Description:
This module implements the complete Legal Hallucination Detector pipeline, orchestrating 
all components to identify factual inaccuracies in legal text. It processes text from input
to final hallucination determination through a series of specialized stages.

Pipeline Stages:
1. Text Partitioning
   - Splits input text into individual paragraphs for analysis
   - Preserves formatting and structure of legal documents

2. First-Stage Classification
   - Filters paragraphs to identify which require fact-checking
   - Uses Aleph-BERT model fine-tuned for Hebrew legal text

3. Search Query Generation
   - Creates optimized search queries for paragraphs needing verification
   - Leverages OpenAI API for targeted legal fact-checking queries

4. Data Retrieval
   - Retrieves evidence from external sources via Google Search API
   - Extracts and processes content from search results

5. Embedding Generation
   - Creates vector representations of paragraphs and search results
   - Uses VoyageAI API for high-quality text embeddings

6. Final Classification
   - Determines presence of hallucinations by comparing embeddings
   - Outputs detailed report with confidence scores and evidence

This module serves as the primary entry point for the Legal Hallucination Detector system,
providing a comprehensive pipeline for detecting factual inaccuracies in legal text.
"""

import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Union, Tuple
import numpy as np
import torch
from dotenv import load_dotenv
# Import Hugging Face authentication
from huggingface_hub import login

# Import modules for paragraph processing
from paragraph_annotation_handler import split_text_to_paragraphs

# Import first-stage classifier (wrapped with mock implementation if needed)
import importlib.util
import sys

# Create mock classifier in case we can't load the real one
class MockFirstClassifier:
    """Mock implementation of the first-stage classifier to handle HuggingFace model access issues."""
    
    @staticmethod
    def predict_needs_check(text):
        """
        Mock function that always predicts the text needs checking with high confidence.
        
        Args:
            text (str): Hebrew text paragraph
            
        Returns:
            tuple: (prediction (always 1), confidence score (always 0.9))
        """
        print("  Using mock first-stage classifier (always returns NEEDS CHECK)")
        return 1, 0.9

# Try to import real classifier, fall back to mock if it fails
try:
    import first_classifier_predictor
    # Test if it's properly initialized
    test_result = first_classifier_predictor.predict_needs_check("Test")
    print("Successfully loaded real first-stage classifier")
except Exception as e:
    print(f"Failed to load real first-stage classifier: {e}")
    print("Using mock first-stage classifier instead")
    # Create a mock module with the same interface
    first_classifier_predictor = MockFirstClassifier
    
# Import pipeline components
from hallucination_detector_search_query_generator import SearchQueryGenerator
from hallucination_detector_search_results_retriever import SearchResultsRetriever
from hallucination_detector_embedding_generator import EmbeddingGenerator

# Import the simple hallucination classifier from the training module
from hallucination_classifier_simple_trainer import HallucinationClassifier

# Load environment variables
load_dotenv()

# Set up Hugging Face authentication if token is available
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
if huggingface_token:
    print("Authenticating with Hugging Face...")
    login(token=huggingface_token)
    print("Hugging Face authentication complete")
else:
    print("Warning: No Hugging Face token found in environment variables. Private models may not be accessible.")


class HallucinationDetector:
    """Main class for detecting hallucinations in legal text."""

    def __init__(self):
        """Initialize the hallucination detector with all required components."""
        # Initialize components
        print("Initializing Legal Hallucination Detector...")
        
        try:
            # Check for required API keys
            required_env_vars = {
                "OPENAI_API_KEY": "OpenAI API key for query generation",
                "SEARCH_API_KEY": "Google Custom Search API key for retrieval",
                "SEARCH_ENGINE_ID": "Google Custom Search Engine ID",
                "API_KEY": "VoyageAI API key for embeddings"
            }
            
            for var_name, description in required_env_vars.items():
                if not os.getenv(var_name):
                    raise ValueError(f"Missing {description} ({var_name}). Please set in .env file.")

            # Initialize first-stage classifier
            print("Loading first-stage classifier...")
            self.first_classifier = first_classifier_predictor

            # Initialize query generator
            print("Initializing search query generator...")
            self.query_generator = SearchQueryGenerator()

            # Initialize search results retriever
            print("Initializing search results retriever...")
            self.search_retriever = SearchResultsRetriever()

            # Initialize embedding generator
            print("Initializing embedding generator...")
            self.hallucination_detector_embedding_generator = EmbeddingGenerator()

            # Initialize final classifier
            print("Loading final hallucination classifier...")
            self.final_classifier = self._load_final_classifier()

            print("Hallucination detector initialized successfully!")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize hallucination detector: {e}")

    def _load_final_classifier(self) -> Any:
        """
        Load the final hallucination classifier model.
        
        Returns:
            torch.nn.Module: The loaded classifier model.
        """
        try:
            # Set device
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
            
            print(f"Using {device} device for final classifier")
            
            # Initialize model architecture (match the architecture in saved model)
            model = HallucinationClassifier(input_dim=2048)  # Changed from 4096 to 2048 to match saved model
            
            # Load saved model weights
            model_paths = [
                # Specific model path
                "hebrew_hallucination_classifier_20250616_152007.pth",
            ]
            
            model_loaded = False
            for path in model_paths:
                if os.path.exists(path):
                    print(f"Loading classifier from {path}")
                    checkpoint = torch.load(path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.to(device)
                    model.eval()  # Set model to evaluation mode
                    model_loaded = True
                    break
            
            if not model_loaded:
                raise FileNotFoundError("Could not find classifier model file.")
                
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load final classifier: {e}")

    def detect_hallucinations(self, input_text: str) -> Dict[str, Any]:
        """
        Process text through the hallucination detection pipeline.
        
        Args:
            input_text (str): The input text to analyze.
            
        Returns:
            dict: Results of the hallucination detection.
        """
        start_time = time.time()
        
        try:
            print("\nBeginning hallucination detection...")
            
            # 1. Split input text into paragraphs
            print("Splitting text into paragraphs...")
            paragraphs = split_text_to_paragraphs(input_text)
            print(f"Split into {len(paragraphs)} paragraphs")
            
            # 2. Process each paragraph
            results = []
            for i, paragraph in enumerate(paragraphs):
                print(f"\nProcessing paragraph {i+1}/{len(paragraphs)}")
                paragraph_result = self._process_paragraph(paragraph, i)
                results.append(paragraph_result)
            
            # 3. Compile final report
            end_time = time.time()
            processing_time = end_time - start_time
            
            report = self._generate_report(results, processing_time)
            return report
            
        except Exception as e:
            error_message = f"Error in hallucination detection pipeline: {e}"
            print(f"ERROR: {error_message}")
            return {"error": error_message}

    def _process_paragraph(self, paragraph: str, paragraph_id: int) -> Dict[str, Any]:
        """
        Process a single paragraph through the pipeline.
        
        Args:
            paragraph (str): The paragraph text to process.
            paragraph_id (int): The ID of the paragraph.
            
        Returns:
            dict: Results of processing the paragraph.
        """
        result = {
            "paragraph_id": paragraph_id,
            "paragraph": paragraph,
            "needs_check": False,
            "hallucination": None,
            "confidence": None,
            "error": None
        }
        
        try:
            # Step 1: First-stage classification
            print("  Running first-stage classification...")
            needs_check, confidence = self.first_classifier.predict_needs_check(paragraph)
            result["needs_check"] = bool(needs_check)
            result["first_stage_confidence"] = float(confidence)
            
            # Skip the rest if paragraph doesn't need checking
            if not needs_check:
                print("  Paragraph does not need checking (confidence: {:.2f}%)".format(confidence * 100))
                return result
            
            print("  Paragraph needs checking (confidence: {:.2f}%)".format(confidence * 100))
            
            # Step 2: Generate search query
            try:
                print("  Generating search query...")
                print("  DEBUG: Starting query generation with OpenAI API...")
                start_time = time.time()
                query = self.query_generator.generate_query(paragraph)
                end_time = time.time()
                print(f"  DEBUG: Query generation took {end_time - start_time:.2f} seconds")
                result["query"] = query
                print(f"  Query: {query}")
                
            except Exception as e:
                error = f"Search query generation failed: {str(e)}"
                print(f"  ERROR: {error}")
                result["error"] = error
                return result
            
            # Step 3: Get search results
            try:
                print("  Retrieving search results...")
                print("  DEBUG: Starting Google search API call...")
                start_time = time.time()
                search_results = self.search_retriever.get_search_results(query)
                end_time = time.time()
                print(f"  DEBUG: Search results retrieval took {end_time - start_time:.2f} seconds")
                result["search_results_count"] = len(search_results)
                print(f"  Retrieved {len(search_results)} search results")
                
                # Include search result titles in the result
                if search_results:
                    result["search_titles"] = [r["title"] for r in search_results]
            
            except Exception as e:
                error = f"Search results retrieval failed: {str(e)}"
                print(f"  ERROR: {error}")
                result["error"] = error
                return result
            
            # Step 4: Generate embeddings
            try:
                print("  Generating embeddings...")
                print("  DEBUG: Starting VoyageAI embedding generation...")
                start_time = time.time()
                embeddings = self.hallucination_detector_embedding_generator.get_embeddings(paragraph, search_results)
                end_time = time.time()
                print(f"  DEBUG: Embedding generation took {end_time - start_time:.2f} seconds")
                print("  DEBUG: Embeddings generated successfully")
                
            except Exception as e:
                error = f"Embedding generation failed: {str(e)}"
                print(f"  ERROR: {error}")
                result["error"] = error
                return result
                
            # Step 5: Final classification
            try:
                print("  Running final classification...")
                hallucination, confidence = self._predict_hallucination(embeddings["combined_features"])
                result["hallucination"] = bool(hallucination)
                result["confidence"] = float(confidence)
                
                print("  Classification complete - " +
                     ("HALLUCINATION" if hallucination else "NO HALLUCINATION") +
                     f" (confidence: {confidence * 100:.2f}%)")
                
            except Exception as e:
                error = f"Final classification failed: {str(e)}"
                print(f"  ERROR: {error}")
                result["error"] = error
                return result
            
        except Exception as e:
            error = f"Error processing paragraph: {str(e)}"
            print(f"  ERROR: {error}")
            result["error"] = error
            
        return result

    def _predict_hallucination(self, features) -> Tuple[bool, float]:
        """
        Predict hallucination using the final classifier.
        
        Args:
            features: Combined features vector.
            
        Returns:
            tuple: (hallucination prediction, confidence)
        """
        # Set device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
            
        # Convert features to tensor
        features_tensor = torch.FloatTensor(features).to(device)
        
        # Ensure correct shape for model input (batch dimension)
        if len(features_tensor.shape) == 1:
            features_tensor = features_tensor.unsqueeze(0)
            
        # Make prediction
        with torch.no_grad():
            outputs = self.final_classifier(features_tensor)
            probabilities = outputs.cpu().numpy()
            
        # Get prediction and confidence
        probability = float(probabilities[0])
        prediction = probability >= 0.5
        
        # Get confidence (distance from decision boundary)
        confidence = max(probability, 1 - probability)
        
        return prediction, confidence

    def _generate_report(self, results: List[Dict[str, Any]], processing_time: float) -> Dict[str, Any]:
        """
        Generate a comprehensive report from detection results.
        
        Args:
            results (list): List of paragraph processing results.
            processing_time (float): Total processing time.
            
        Returns:
            dict: Complete report.
        """
        # Calculate summary statistics
        total_paragraphs = len(results)
        needing_check = sum(1 for r in results if r["needs_check"])
        
        # Count hallucinations among paragraphs that needed checking
        hallucinations = sum(1 for r in results if r["needs_check"] and r.get("hallucination") is True)
        
        # Count errors
        errors = sum(1 for r in results if r.get("error") is not None)
        
        # Calculate overall hallucination rate
        hallucination_rate = hallucinations / total_paragraphs if total_paragraphs > 0 else 0
        
        # Create report
        report = {
            "timestamp": datetime.now().isoformat(),
            "processing_time_seconds": processing_time,
            "summary": {
                "total_paragraphs": total_paragraphs,
                "paragraphs_needing_check": needing_check,
                "paragraphs_with_hallucinations": hallucinations,
                "overall_hallucination_rate": hallucination_rate,
                "processing_errors": errors
            },
            "detailed_results": results
        }
        
        # Generate text report
        text_report = self._format_text_report(report)
        report["text_report"] = text_report
        
        return report

    def _format_text_report(self, report: Dict[str, Any]) -> str:
        """
        Format the report as a simple text document.
        
        Args:
            report (dict): The report data.
            
        Returns:
            str: Formatted text report.
        """
        summary = report["summary"]
        results = report["detailed_results"]
        
        # Create header
        header = "=" * 50 + "\n"
        header += "LEGAL HALLUCINATION DETECTION REPORT\n"
        header += "=" * 50 + "\n"
        header += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"Processing time: {report['processing_time_seconds']:.2f} seconds\n\n"
        
        # Create summary section
        summary_text = "SUMMARY:\n"
        summary_text += f"- Total paragraphs analyzed: {summary['total_paragraphs']}\n"
        summary_text += f"- Paragraphs needing verification: {summary['paragraphs_needing_check']}/{summary['total_paragraphs']} "
        summary_text += f"({summary['paragraphs_needing_check']/summary['total_paragraphs']*100:.1f}%)\n"
        
        if summary['paragraphs_needing_check'] > 0:
            summary_text += f"- Paragraphs with hallucinations: {summary['paragraphs_with_hallucinations']}/{summary['paragraphs_needing_check']} "
            summary_text += f"({summary['paragraphs_with_hallucinations']/summary['paragraphs_needing_check']*100:.1f}%)\n"
        
        summary_text += f"- Overall hallucination rate: {summary['overall_hallucination_rate']*100:.1f}%\n"
        
        if summary['processing_errors'] > 0:
            summary_text += f"- Processing errors: {summary['processing_errors']}\n"
        
        # Create detailed results section
        details = "\nDETAILS:\n"
        for result in results:
            details += "-" * 50 + "\n"
            details += f"[{result['paragraph_id'] + 1}] "
            
            if result.get("error"):
                details += "⚠️ ERROR\n"
                details += f"    Error: {result['error']}\n"
                details += f"    Paragraph: {result['paragraph'][:100]}...\n"
                continue
                
            if not result["needs_check"]:
                details += "⚪ NO CHECK NEEDED "
                details += f"(Confidence: {result['first_stage_confidence']*100:.1f}%)\n"
                details += f"    {result['paragraph'][:200]}...\n" if len(result['paragraph']) > 200 else f"    {result['paragraph']}\n"
                continue
                
            if result["hallucination"]:
                details += "❌ HALLUCINATION DETECTED "
                details += f"(Confidence: {result['confidence']*100:.1f}%)\n"
            else:
                details += "✓ VERIFIED "
                details += f"(Confidence: {result['confidence']*100:.1f}%)\n"
                
            details += f"    {result['paragraph'][:200]}...\n" if len(result['paragraph']) > 200 else f"    {result['paragraph']}\n"
            details += f"    Query: \"{result['query']}\"\n"
            
            if result.get("search_titles"):
                details += f"    Top search results: {', '.join(result['search_titles'][:3])}\n"
        
        # Combine all sections
        full_report = header + summary_text + details + "\n" + "=" * 50
        return full_report


def process_text_file(file_path):
    """
    Process text from a file through the hallucination detector.
    
    Args:
        file_path (str): Path to text file to process.
        
    Returns:
        dict: Detection results.
    """
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Initialize detector
        detector = HallucinationDetector()
        
        # Process the text
        results = detector.detect_hallucinations(text)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"hallucination_report_{timestamp}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(results["text_report"])
            
        print(f"\nReport saved to {output_file}")
        
        # Print summary to console
        print("\n" + results["text_report"])
        
        return results
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return {"error": str(e)}


def main():
    """
    Main entry point for the hallucination detector.
    """
    # Load environment variables
    load_dotenv()
    
    print("Legal Hallucination Detector")
    print("=" * 50)
    
    # Check command line arguments for file input
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"Processing file: {file_path}")
        process_text_file(file_path)
        return
    
    # Interactive mode
    print("\nEnter text to check for hallucinations (or type 'exit' to quit):")
    print("For multi-line input, press Enter twice to finish.")
    
    try:
        while True:
            print("\n" + "-" * 50)
            print("Enter text (empty line to finish, 'exit' to quit):")
            
            lines = []
            while True:
                line = input()
                if not line:
                    break
                if line.lower() == 'exit':
                    print("Exiting...")
                    return
                lines.append(line)
                
            if not lines:
                continue
                
            text = "\n".join(lines)
            
            # Initialize detector
            detector = HallucinationDetector()
            
            # Process the text
            results = detector.detect_hallucinations(text)
            
            # Print report
            print("\n" + results["text_report"])
            
            # Ask if user wants to save the report
            save = input("\nSave report to file? (y/n): ").lower().strip()
            if save == 'y':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"hallucination_report_{timestamp}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(results["text_report"])
                print(f"Report saved to {output_file}")
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
