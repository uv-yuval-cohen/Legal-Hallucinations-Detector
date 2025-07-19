"""
Module: hallucination_detector_search_query_generator.py

Description:
This module generates optimized search queries for legal paragraphs using the OpenAI API.
It's a component of the Legal Hallucination Detector pipeline, transforming paragraphs 
that need verification into effective search queries that can retrieve relevant evidence.

Key Functionality:
1. OpenAI Integration
   - Leverages the OpenAI API with carefully engineered prompts
   - Optimized for legal text analysis and query generation
   - Supports customizable temperature and prompt engineering

2. Error Handling
   - Robust handling of API rate limits and timeouts
   - Graceful degradation with informative error messages
   - Retry mechanism for transient failures

3. Query Optimization
   - Extracts key legal concepts and claims from paragraphs
   - Formats queries for maximum search effectiveness
   - Supports Hebrew and multilingual text

This module bridges the gap between paragraph identification and evidence retrieval
in the hallucination detection pipeline.
"""

import os
import time
import json
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

class SearchQueryGenerator:
    """Class to generate search queries from paragraphs using OpenAI."""

    def __init__(self, api_key=None, model="gpt-3.5-turbo"):
        """
        Initialize the search query generator.

        Args:
            api_key (str, optional): OpenAI API key. If None, loads from environment variable.
            model (str, optional): OpenAI model to use. Defaults to "gpt-3.5-turbo".
        """
        # Set API key from parameter or environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please provide it or set OPENAI_API_KEY environment variable.")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model
        
        # Default system prompt for query generation
        self.system_prompt = """
        You are a specialized legal search query generator. Your task is to create effective search queries for fact-checking legal claims.
        
        Guidelines:
        1. Focus on factual claims, legal precedents, or assertions that need verification
        2. Create concise, focused queries (1-3 search terms)
        3. Prioritize specific legal terms, case references, or statutory citations
        4. For Hebrew text, generate queries in Hebrew
        5. Do NOT explain your reasoning, only output the search query
        """

    def generate_query(self, paragraph, temperature=0.3, custom_prompt=None):
        """
        Generate a search query for a paragraph.

        Args:
            paragraph (str): The paragraph text to generate a query for.
            temperature (float, optional): Temperature parameter for OpenAI API. Defaults to 0.3.
            custom_prompt (str, optional): Custom system prompt. If None, uses default.

        Returns:
            str: Generated search query.

        Raises:
            Exception: If query generation fails after retries.
        """
        system_prompt = custom_prompt or self.system_prompt
        
        # User prompt with the paragraph
        user_prompt = f"""
        Generate a concise search query to fact-check the following legal paragraph. 
        Focus on the most important factual claims that need verification:

        {paragraph}

        SEARCH QUERY:
        """
        
        # Try API call first, fall back to simple extraction if it fails
        try:
            # Retry mechanism for API calls
            for attempt in range(MAX_RETRIES):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=temperature,
                        max_tokens=100
                    )
                    
                    query = response.choices[0].message.content.strip()
                    return query
                    
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
                        raise Exception(f"Query generation failed after {MAX_RETRIES} attempts: {e}")
                    
                    # Wait before retry for other errors
                    time.sleep(RETRY_DELAY)
            
            # This should not be reached but just in case
            raise Exception(f"Query generation failed after {MAX_RETRIES} attempts")
            
        except Exception as api_error:
            # Fallback to simple keyword extraction
            print(f"Warning: API query generation failed ({api_error}). Using simple keyword extraction fallback.")
            
            # Get first 5-10 words as a simple query
            words = paragraph.split()
            if len(words) <= 5:
                query_words = words
            else:
                # Get first 5 words
                query_words = words[:5]
                
            query = " ".join(query_words)
            return query

    def generate_query_batch(self, paragraphs, temperature=0.3, custom_prompt=None):
        """
        Generate search queries for multiple paragraphs.

        Args:
            paragraphs (list): List of paragraph texts.
            temperature (float, optional): Temperature parameter for OpenAI API. Defaults to 0.3.
            custom_prompt (str, optional): Custom system prompt. If None, uses default.

        Returns:
            list: List of generated search queries.
        """
        queries = []
        for i, paragraph in enumerate(paragraphs):
            print(f"Generating query for paragraph {i+1}/{len(paragraphs)}")
            try:
                query = self.generate_query(paragraph, temperature, custom_prompt)
                queries.append(query)
                # Add small delay between requests
                if i < len(paragraphs) - 1:
                    time.sleep(1)
            except Exception as e:
                print(f"Failed to generate query for paragraph {i+1}: {e}")
                queries.append(None)
                
        return queries


def main():
    """
    Test the search query generator with example paragraphs.
    """
    # Load API key from environment
    load_dotenv()
    
    # Example paragraphs for testing
    test_paragraphs = [
        "חוק הגנת הצרכן אוסר על הטעיית צרכנים, וקובע כי על עוסק לגלות כל פרט מהותי לעסקה.",
        "בפסק דין תקדימי, קבע בית המשפט העליון כי חוזה אחיד יכול להיות בטל אם יש בו תנאי מקפח כלפי הצד החלש.",
    ]
    
    try:
        # Initialize generator
        generator = SearchQueryGenerator()
        
        print("Testing query generation...")
        for i, paragraph in enumerate(test_paragraphs):
            print(f"\nParagraph {i+1}: {paragraph[:100]}...")
            
            query = generator.generate_query(paragraph)
            print(f"Generated query: {query}")
            
        print("\nQuery generator testing complete!")
        
    except Exception as e:
        print(f"Error in test run: {e}")


if __name__ == "__main__":
    main()
