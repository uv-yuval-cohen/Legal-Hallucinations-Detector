import json
import re
import time
import numpy as np
import voyageai
from typing import List, Tuple, Optional
import logging

from dotenv import load_dotenv
from tqdm import tqdm
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding_conversion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VoyageEmbeddingConverter:
    def __init__(self, api_key: Optional[str] = None, model: str = "voyage-3-large"):
        """
        Initialize Voyage AI client.
        If api_key is None, will use VOYAGE_API_KEY environment variable.
        """
        self.model = model
        if api_key:
            self.client = voyageai.Client(api_key=api_key)
        else:
            self.client = voyageai.Client()  # Uses VOYAGE_API_KEY env var
        self.embedding_dim = 1024

    def extract_text_from_formatted(self, formatted_text: str) -> Tuple[str, str]:
        """
        Extract clean original and search text from formatted string.

        Format: "[ORIGINAL] text [/ORIGINAL] [SEARCH] text [/SEARCH]"
        Returns: (original_text, search_text)
        """
        # Extract original text
        original_match = re.search(r'\[ORIGINAL\](.*?)\[/ORIGINAL\]', formatted_text, re.DOTALL)
        original_text = original_match.group(1).strip() if original_match else ""

        # Extract search text
        search_match = re.search(r'\[SEARCH\](.*?)\[/SEARCH\]', formatted_text, re.DOTALL)
        search_text = search_match.group(1).strip() if search_match else ""

        return original_text, search_text

    def get_embeddings_batch(self, texts: List[str], max_retries: int = 3) -> Optional[List[List[float]]]:
        """
        Get embeddings for a batch of texts using Voyage AI API.
        """
        if not texts or all(not text.strip() for text in texts):
            logger.warning("Empty texts provided to batch embedding")
            return None

        for attempt in range(max_retries):
            try:
                result = self.client.embed(
                    texts=texts,
                    model=self.model,
                    input_type="document"
                )
                return result.embeddings

            except Exception as e:
                error_msg = str(e).lower()

                if "rate limit" in error_msg or "429" in error_msg:
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}")
                    time.sleep(wait_time)
                    continue

                logger.error(f"API error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(1)

        return None

    def process_batch(self, batch_data: List[dict], start_idx: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Process a batch of data and return embeddings arrays.
        """
        batch_size = len(batch_data)
        original_embeddings = np.zeros((batch_size, self.embedding_dim))
        search_embeddings = np.zeros((batch_size, self.embedding_dim))
        labels = np.zeros(batch_size, dtype=int)
        indices = np.zeros(batch_size, dtype=int)

        # Extract all texts first
        original_texts = []
        search_texts = []
        valid_indices = []

        for i, row in enumerate(batch_data):
            try:
                original_text, search_text = self.extract_text_from_formatted(row['text'])

                if not original_text or not search_text:
                    logger.warning(f"Row {row.get('index', start_idx + i)}: Empty text after extraction")
                    continue

                original_texts.append(original_text)
                search_texts.append(search_text)
                valid_indices.append(i)

            except Exception as e:
                logger.error(f"Error extracting text from row {row.get('index', start_idx + i)}: {e}")
                continue

        if not original_texts:
            logger.warning("No valid texts found in batch")
            return original_embeddings, search_embeddings, labels, indices

        logger.info(f"Getting embeddings for {len(original_texts)} original texts...")

        # Get embeddings in batches (API can handle multiple texts at once)
        original_embeddings_list = self.get_embeddings_batch(original_texts)
        if original_embeddings_list is None:
            logger.error("Failed to get original embeddings for batch")
            return original_embeddings, search_embeddings, labels, indices

        logger.info(f"Getting embeddings for {len(search_texts)} search texts...")

        search_embeddings_list = self.get_embeddings_batch(search_texts)
        if search_embeddings_list is None:
            logger.error("Failed to get search embeddings for batch")
            return original_embeddings, search_embeddings, labels, indices

        # Store results for valid entries
        for j, i in enumerate(valid_indices):
            original_embeddings[i] = np.array(original_embeddings_list[j])
            search_embeddings[i] = np.array(search_embeddings_list[j])
            labels[i] = batch_data[i]['label']
            indices[i] = batch_data[i].get('index', start_idx + i)

        valid_count = len(valid_indices)
        logger.info(f"Successfully processed {valid_count}/{batch_size} rows in this batch")

        return original_embeddings, search_embeddings, labels, indices

    def convert_dataset(self, input_file: str, output_file: str = "hebrew_hallucination_embeddings.npz",
                        batch_size: int = 20, resume_from: int = 0):
        """
        Convert entire dataset from JSONL to embeddings and save as NPZ.
        """
        logger.info(f"Starting conversion of {input_file}")
        logger.info(f"Using model: {self.model}")
        logger.info(f"Batch size: {batch_size}")

        # Load data
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))

        total_rows = len(data)
        logger.info(f"Loaded {total_rows} rows")

        # Resume from specific point if needed
        if resume_from > 0:
            data = data[resume_from:]
            logger.info(f"Resuming from row {resume_from}")

        # Initialize result arrays
        all_original_embeddings = []
        all_search_embeddings = []
        all_labels = []
        all_indices = []

        # Process in batches
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            current_idx = resume_from + i

            logger.info(f"Processing batch {i // batch_size + 1}/{(len(data) + batch_size - 1) // batch_size} "
                        f"(rows {current_idx} to {current_idx + len(batch) - 1})")

            try:
                orig_emb, search_emb, labels, indices = self.process_batch(batch, current_idx)

                # Filter out zero rows (failed embeddings)
                valid_mask = (orig_emb.sum(axis=1) != 0) & (search_emb.sum(axis=1) != 0)

                if valid_mask.sum() > 0:
                    all_original_embeddings.append(orig_emb[valid_mask])
                    all_search_embeddings.append(search_emb[valid_mask])
                    all_labels.append(labels[valid_mask])
                    all_indices.append(indices[valid_mask])
                else:
                    logger.warning(f"No valid embeddings in this batch")

                # Save intermediate results every few batches
                if (i // batch_size + 1) % 5 == 0:
                    self._save_intermediate_results(all_original_embeddings, all_search_embeddings,
                                                    all_labels, all_indices, f"{output_file}.tmp")

                # Small delay between batches to be nice to the API
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                continue

        # Combine all results
        if all_original_embeddings:
            final_original = np.vstack(all_original_embeddings)
            final_search = np.vstack(all_search_embeddings)
            final_labels = np.concatenate(all_labels)
            final_indices = np.concatenate(all_indices)

            # Save final results
            np.savez_compressed(
                output_file,
                original_embeddings=final_original,
                search_embeddings=final_search,
                labels=final_labels,
                indices=final_indices
            )

            logger.info(f"Successfully saved {len(final_labels)} samples to {output_file}")
            logger.info(f"Original embeddings shape: {final_original.shape}")
            logger.info(f"Search embeddings shape: {final_search.shape}")
            logger.info(f"Labels distribution: {np.bincount(final_labels)}")

            # Clean up temporary file
            temp_file = f"{output_file}.tmp"
            if os.path.exists(temp_file):
                os.remove(temp_file)

        else:
            logger.error("No valid embeddings were generated!")

    def _save_intermediate_results(self, orig_emb_list, search_emb_list, labels_list, indices_list, filename):
        """Save intermediate results to prevent data loss."""
        if orig_emb_list:
            try:
                temp_original = np.vstack(orig_emb_list)
                temp_search = np.vstack(search_emb_list)
                temp_labels = np.concatenate(labels_list)
                temp_indices = np.concatenate(indices_list)

                np.savez_compressed(
                    filename,
                    original_embeddings=temp_original,
                    search_embeddings=temp_search,
                    labels=temp_labels,
                    indices=temp_indices
                )
                logger.info(f"Saved intermediate results: {len(temp_labels)} samples")
            except Exception as e:
                logger.error(f"Failed to save intermediate results: {e}")


def main():
    # Configuration
    # Load environment variables from .env file
    load_dotenv()

    API_KEY = os.getenv('API_KEY')
    if not API_KEY:
        raise ValueError("API_KEY environment variable is not set")  # Set to your API key, or None to use VOYAGE_API_KEY env var

    INPUT_FILE = "michael_evaluation_searchResults.jsonl"
    OUTPUT_FILE = "evaluation_pack_embeddings.npz"
    BATCH_SIZE = 4
    RESUME_FROM = 0  # Set to > 0 if resuming from a specific row

    # Initialize converter
    converter = VoyageEmbeddingConverter(api_key=API_KEY)

    # Run conversion
    converter.convert_dataset(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        batch_size=BATCH_SIZE,
        resume_from=RESUME_FROM
    )

    logger.info("Conversion complete!")


if __name__ == "__main__":
    main()