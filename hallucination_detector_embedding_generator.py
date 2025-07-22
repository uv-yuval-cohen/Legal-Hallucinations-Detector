"""
Module: hallucination_detector_embedding_generator.py

Description:
This module generates embeddings for paragraphs and search results using the VoyageAI API.
It's a key component of the Legal Hallucination Detector pipeline, transforming text into 
vector representations that can be processed by the final classifier.

Key Functionality:
1. VoyageAI Integration
   - Leverages the VoyageAI API for high-quality embeddings
   - Supports processing of both original paragraphs and search results
   - Optimized for Hebrew legal text

2. Text Formatting
   - Prepares text with appropriate markers for embedding generation
   - Handles concatenation of multiple search results
   - Ensures consistent input structure for the classifier

3. Error Handling
   - Robust API error handling with retries
   - Graceful degradation with informative error messages
   - Memory-efficient processing of large texts

This module serves as the bridge between raw text and numerical representations
that can be analyzed by the final hallucination classifier.
"""

import os
import time
import numpy as np
import voyageai
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

# Configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
EMBEDDING_DIM = 1024  # Voyage-3-large embedding dimension

class EmbeddingGenerator:
    """Class to generate embeddings using VoyageAI API."""

    def __init__(self, api_key=None, model="voyage-3-large", output_dimension=2048):
        """
        Initialize the embedding generator.

        Args:
            api_key (str, optional): VoyageAI API key. If None, loads from environment variable.
            model (str, optional): VoyageAI model to use. Defaults to "voyage-3-large".
            output_dimension (int, optional): Dimension of the combined feature output. 
                                             Use 2048 for simple_model and 4096 for k_fold_model.
        """
        # Set API key from parameter or environment
        self.api_key = api_key or os.getenv("API_KEY")
        if not self.api_key:
            raise ValueError("VoyageAI API key not found. Please provide it or set API_KEY environment variable.")
        
        self.model = model
        self.client = voyageai.Client(api_key=self.api_key)
        self.embedding_dim = EMBEDDING_DIM
        self.output_dimension = output_dimension
        
        print(f"Embedding generator initialized with output dimension: {self.output_dimension}")
        
    def get_embeddings(self, paragraph: str, search_results: List[Dict]) -> Dict:
        """
        Generate embeddings for paragraph and search results.

        Args:
            paragraph (str): The original paragraph text.
            search_results (list): List of search result dictionaries.

        Returns:
            dict: Dictionary containing original and search embeddings.

        Raises:
            Exception: If embedding generation fails after retries.
        """
        # Verify inputs
        if not paragraph.strip():
            raise ValueError("Empty paragraph provided")
            
        # Extract full text from search results
        search_texts = []
        for result in search_results:
            if "full_text" in result and result["full_text"].strip():
                search_texts.append(result["full_text"])
        
        # Handle case with no valid search results
        if not search_texts:
            print("Warning: No valid search results with text. Using empty string.")
            search_text = ""
        else:
            # Combine search results with separators
            search_text = "\n\n".join(search_texts)
        
        # Format texts with markers for embedding generation
        formatted_text = f"[ORIGINAL] {paragraph} [/ORIGINAL] [SEARCH] {search_text} [/SEARCH]"
        
        # Generate embeddings for the combined text
        embedding_dict = self._generate_embeddings(formatted_text)
        
        return embedding_dict
    
    def _generate_embeddings(self, formatted_text: str) -> Dict:
        """
        Generate embeddings for formatted text using VoyageAI API.

        Args:
            formatted_text (str): Formatted text with markers.

        Returns:
            dict: Dictionary containing original and search embeddings.

        Raises:
            Exception: If embedding generation fails after retries.
        """
        # Retry mechanism for API calls
        for attempt in range(MAX_RETRIES):
            try:
                # Extract original and search text for separate embeddings
                original_text, search_text = self._extract_text_from_formatted(formatted_text)
                
                # Get embeddings for original text
                original_result = self.client.embed(
                    texts=[original_text],
                    model=self.model,
                    input_type="document"
                )
                original_embedding = np.array(original_result.embeddings[0])
                
                # Get embeddings for search text (handle empty search text)
                if not search_text.strip():
                    # Use zeros for empty search text
                    search_embedding = np.zeros(self.embedding_dim)
                else:
                    search_result = self.client.embed(
                        texts=[search_text],
                        model=self.model,
                        input_type="document"
                    )
                    search_embedding = np.array(search_result.embeddings[0])
                
                # Create feature vector combinations as expected by the classifier
                element_product = original_embedding * search_embedding
                
                # Combine features based on the required output dimension
                if self.output_dimension == 2048:
                    # Original + search embeddings (2048 dims total)
                    features = np.concatenate([
                        original_embedding,          # 1024 dims
                        search_embedding             # 1024 dims
                    ])                               # Total: 2048 dims
                elif self.output_dimension == 4096:
                    # Original + search + element-wise product + difference (4096 dims total)
                    # Additional features for k_fold model
                    difference = original_embedding - search_embedding
                    features = np.concatenate([
                        original_embedding,          # 1024 dims
                        search_embedding,            # 1024 dims
                        element_product,             # 1024 dims
                        difference                   # 1024 dims
                    ])                               # Total: 4096 dims
                else:
                    # Default to 2048 dimensions if unsupported dimension is requested
                    print(f"Warning: Unsupported output dimension {self.output_dimension}, defaulting to 2048")
                    features = np.concatenate([
                        original_embedding,
                        search_embedding
                    ])
                
                # Create embedding dictionary
                embedding_dict = {
                    "original_embedding": original_embedding.tolist(),
                    "search_embedding": search_embedding.tolist(),
                    "combined_features": features.tolist()
                }
                
                return embedding_dict
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Handle rate limits with exponential backoff
                if "rate limit" in error_msg or "429" in error_msg:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    print(f"Rate limit hit. Waiting {wait_time}s before retry {attempt+1}/{MAX_RETRIES}")
                    time.sleep(wait_time)
                    continue
                    
                # Re-raise on last attempt
                if attempt == MAX_RETRIES - 1:
                    raise Exception(f"Embedding generation failed after {MAX_RETRIES} attempts: {e}")
                
                # Wait before retry for other errors
                time.sleep(RETRY_DELAY)
        
        raise Exception(f"Embedding generation failed after {MAX_RETRIES} attempts")
    
    def _extract_text_from_formatted(self, formatted_text: str) -> Tuple[str, str]:
        """
        Extract original and search text from formatted text.

        Args:
            formatted_text (str): Text with [ORIGINAL] and [SEARCH] markers.

        Returns:
            tuple: (original_text, search_text)
        """
        import re
        
        # Extract original text
        original_match = re.search(r'\[ORIGINAL\](.*?)\[/ORIGINAL\]', formatted_text, re.DOTALL)
        original_text = original_match.group(1).strip() if original_match else ""

        # Extract search text
        search_match = re.search(r'\[SEARCH\](.*?)\[/SEARCH\]', formatted_text, re.DOTALL)
        search_text = search_match.group(1).strip() if search_match else ""

        return original_text, search_text


def main():
    """
    Test the embedding generator with example text.
    """
    # Load environment variables
    load_dotenv()
    
    # Example text and search results for testing
    test_paragraph = "חוק הגנת הצרכן אוסר על הטעיית צרכנים, וקובע כי על עוסק לגלות כל פרט מהותי לעסקה."
    test_search_results = [
        {
            "title": "חוק הגנת הצרכן, תשמ״א-1981",
            "link": "https://www.nevo.co.il/law_html/law01/089_001.htm",
            "snippet": "סעיף 2 לחוק הגנת הצרכן אוסר על הטעיית צרכנים",
            "full_text": "חוק הגנת הצרכן, תשמ״א-1981 קובע כי: איסור הטעיה - לא יעשה עוסק דבר העלול להטעות צרכן בכל ענין מהותי בעסקה."
        }
    ]
    
    try:
        # Initialize generator
        generator = EmbeddingGenerator()
        
        print("Testing embedding generation...")
        print(f"Paragraph: {test_paragraph}")
        print(f"Search results: {len(test_search_results)} results")
        
        embedding_dict = generator.get_embeddings(test_paragraph, test_search_results)
        
        print("\nEmbedding generation successful!")
        print(f"Original embedding shape: {len(embedding_dict['original_embedding'])}")
        print(f"Search embedding shape: {len(embedding_dict['search_embedding'])}")
        print(f"Combined features shape: {len(embedding_dict['combined_features'])}")
        
    except Exception as e:
        print(f"Error in test run: {e}")


if __name__ == "__main__":
    main()
