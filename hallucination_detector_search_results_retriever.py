"""
Module: hallucination_detector_search_results_retriever.py

Description:
This module retrieves search results for verification queries in the Legal Hallucination 
Detector pipeline. It uses the Google Custom Search API to find relevant information 
for fact-checking legal text paragraphs.

Key Functionality:
1. Google Search Integration
   - Leverages the Google Custom Search API for accurate results
   - Optimized for legal and Hebrew content retrieval
   - Returns structured search results with full text extraction

2. Content Extraction
   - Extracts clean text from search results using multiple methods
   - Supports various content types including HTML and PDF documents
   - Handles encoding issues common with Hebrew text

3. Error Handling
   - Robust handling of API limits and network failures
   - Graceful degradation when extraction fails
   - Comprehensive logging for troubleshooting

This module serves as the evidence gathering component in the hallucination detection
pipeline, retrieving the information needed for verification of legal claims.
"""

import os
import time
import requests
import re
from bs4 import BeautifulSoup
from newspaper import Article
from dotenv import load_dotenv

# Try to import PDF processing libraries
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    try:
        import pdfplumber
        PDF_SUPPORT = True
    except ImportError:
        PDF_SUPPORT = False
        print("Warning: No PDF support. Install PyPDF2 or pdfplumber for PDF extraction")

# Configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
MAX_RESULTS = 5  # Number of results to retrieve per query

class SearchResultsRetriever:
    """Class to retrieve search results using Google Custom Search API."""

    def __init__(self, api_key=None, cx=None):
        """
        Initialize the search results retriever.

        Args:
            api_key (str, optional): Google Custom Search API key. If None, loads from environment variable.
            cx (str, optional): Google Custom Search Engine ID. If None, loads from environment variable.
        """
        # Set API key and CX from parameters or environment
        self.api_key = api_key or os.getenv("SEARCH_API_KEY")
        self.cx = cx or os.getenv("SEARCH_ENGINE_ID")
        
        if not self.api_key:
            raise ValueError("Google Search API key not found. Please provide it or set SEARCH_API_KEY environment variable.")
        if not self.cx:
            raise ValueError("Google Custom Search Engine ID not found. Please provide it or set SEARCH_ENGINE_ID environment variable.")
        
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "he,en;q=0.9",
            "Accept-Charset": "utf-8"
        }

    def get_search_results(self, query, max_results=MAX_RESULTS):
        """
        Retrieve search results for a query.

        Args:
            query (str): The search query.
            max_results (int, optional): Maximum number of results to retrieve. Defaults to MAX_RESULTS.

        Returns:
            list: List of dictionaries containing search results with title, link, snippet and extracted text.

        Raises:
            Exception: If search fails after retries.
        """
        if not query or not query.strip():
            raise ValueError("Empty search query provided")
            
        # Step 1: Perform the Google Search API query
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": query,
            "key": self.api_key,
            "cx": self.cx,
            "num": 10,  # Get 10 results to have backup options
            "lr": "lang_he",  # Prefer Hebrew results
            "hl": "he"  # Hebrew interface language
        }
        
        print("\n=== DEBUG: SEARCH REQUEST DETAILS ===")
        print(f"Search URL: {url}")
        print(f"Query: {query}")
        print(f"API Key (first 5 chars): {self.api_key[:5]}...")
        print(f"CX ID (first 5 chars): {self.cx[:5]}...")
        print(f"Full Params: {params}")

        # Retry mechanism for API calls
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                search_data = response.json()
                
                print("\n=== DEBUG: SEARCH API RESPONSE ===")
                print(f"Response status code: {response.status_code}")
                print(f"Response headers: {response.headers}")
                print(f"Response body keys: {search_data.keys()}")
                
                # Print a subset of the response for debugging
                if "searchInformation" in search_data:
                    print(f"Search info: {search_data['searchInformation']}")
                if "queries" in search_data:
                    print(f"Query info: {search_data['queries'].keys()}")
                
                # Check if we have results
                if "items" not in search_data:
                    print(f"No search results found for query: {query}")
                    print("Full response (truncated):")
                    print(str(search_data)[:500])
                    return []
                    
                results = search_data["items"]
                break
                
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
                    print(f"\n=== DEBUG: SEARCH ERROR DETAILS ===")
                    print(f"Error type: {type(e)}")
                    print(f"Error message: {str(e)}")
                    print(f"Attempt: {attempt+1}/{MAX_RETRIES}")
                    if hasattr(e, 'response') and e.response:
                        print(f"Response status code: {e.response.status_code}")
                        print(f"Response headers: {e.response.headers}")
                        try:
                            print(f"Response content: {e.response.text[:500]}...")
                        except:
                            print("Could not access response content")
                    raise Exception(f"Search failed after {MAX_RETRIES} attempts: {e}")
                
                # Wait before retry for other errors
                time.sleep(RETRY_DELAY)
        else:
            # This executes if the for loop completes without a break
            raise Exception(f"Search failed after {MAX_RETRIES} attempts")

        # Step 2: Format initial results
        all_results = [
            {
                "title": item["title"],
                "link": item["link"],
                "snippet": item.get("snippet", ""),
                "full_text": ""
            }
            for item in results
        ]

        # Step 3: Extract article text from each result
        successful_results = []

        for result in all_results:
            if len(successful_results) >= max_results:
                break

            # Check if it's a PDF
            is_pdf = result["link"].lower().endswith('.pdf')

            if is_pdf:
                print(f"Processing PDF: {result['link']}")
                full_text = self._extract_pdf_text(result["link"])
                if full_text and len(full_text.strip()) > 100:
                    full_text = self._clean_text(full_text)
                    # Limit to 5000 characters
                    if len(full_text) > 5000:
                        full_text = full_text[:5000] + "..."
                    result["full_text"] = full_text
                    successful_results.append(result)
                    time.sleep(0.5)
                    continue
                else:
                    print(f"PDF extraction failed or insufficient content")
                    continue

            # Skip other non-HTML files
            if result["link"].lower().endswith(('.doc', '.docx', '.xls', '.xlsx')):
                print(f"Skipping non-HTML file: {result['link']}")
                continue

            try:
                # Use newspaper3k to extract the article
                article = Article(result["link"], language='he')
                article.download()
                article.parse()

                # Get the main article text
                full_text = self._clean_text(article.text)
                if full_text and len(full_text.strip()) > 100:
                    # Limit to 5000 characters
                    if len(full_text) > 5000:
                        full_text = full_text[:5000] + "..."
                    result["full_text"] = full_text
                    successful_results.append(result)
                    continue

            except Exception as e:
                # Fallback: Try basic BeautifulSoup extraction
                try:
                    response = requests.get(result["link"], headers=self.headers, timeout=15)
                    response.raise_for_status()

                    if response.encoding != 'utf-8':
                        response.encoding = 'utf-8'

                    soup = BeautifulSoup(response.text, "html.parser")

                    # Remove script and style elements
                    for script in soup(["script", "style", "nav", "footer", "header"]):
                        script.decompose()

                    # Try to find main content areas
                    main_content = soup.find(['main', 'article', 'div[class*="content"]'])
                    if main_content:
                        paragraphs = main_content.find_all("p")
                    else:
                        paragraphs = soup.find_all("p")

                    full_text = " ".join(para.get_text(strip=True) for para in paragraphs if para.get_text(strip=True))
                    full_text = self._clean_text(full_text)

                    if full_text and len(full_text.strip()) > 100:
                        # Limit to 5000 characters
                        if len(full_text) > 5000:
                            full_text = full_text[:5000] + "..."
                        result["full_text"] = full_text
                        successful_results.append(result)

                except Exception as fallback_e:
                    print(f"Failed to extract from {result['link']}: {e}")
                    continue

            # Add small delay to be respectful to servers
            time.sleep(0.5)

        return successful_results

    def _extract_pdf_text(self, url):
        """
        Extract text from PDF files.

        Args:
            url (str): URL of the PDF file.

        Returns:
            str: Extracted text or empty string if extraction fails.
        """
        if not PDF_SUPPORT:
            return ""

        try:
            response = requests.get(url, headers=self.headers, timeout=20)
            response.raise_for_status()

            # Try PyPDF2 first
            try:
                import PyPDF2
                from io import BytesIO

                pdf_file = BytesIO(response.content)
                reader = PyPDF2.PdfReader(pdf_file)

                text_parts = []
                # Extract from first 5 pages max to avoid too much content
                max_pages = min(5, len(reader.pages))

                for page_num in range(max_pages):
                    page = reader.pages[page_num]
                    text_parts.append(page.extract_text())

                full_text = " ".join(text_parts)
                return full_text.strip()

            except ImportError:
                # Try pdfplumber as fallback
                import pdfplumber
                from io import BytesIO

                pdf_file = BytesIO(response.content)
                text_parts = []

                with pdfplumber.open(pdf_file) as pdf:
                    # Extract from first 5 pages max
                    max_pages = min(5, len(pdf.pages))

                    for page_num in range(max_pages):
                        page = pdf.pages[page_num]
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)

                full_text = " ".join(text_parts)
                return full_text.strip()

        except Exception as e:
            print(f"PDF extraction failed for {url}: {e}")
            return ""

    def _clean_text(self, text):
        """
        Clean extracted text from HTML artifacts and improve readability.

        Args:
            text (str): Text to clean.

        Returns:
            str: Cleaned text.
        """
        if not text:
            return ""

        # Remove excessive whitespace and newlines
        text = re.sub(r'\s+', ' ', text)

        # Remove common HTML artifacts
        text = re.sub(r'[\u200b-\u200d\ufeff]', '', text)  # Remove zero-width characters

        # Remove repeated punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', '---', text)

        # Clean up spacing around Hebrew text
        text = re.sub(r'\s+([א-ת])', r' \1', text)
        text = re.sub(r'([א-ת])\s+', r'\1 ', text)

        return text.strip()


def main():
    """
    Test the search results retriever with example queries.
    """
    # Load environment variables
    load_dotenv()
    
    # Example queries for testing
    test_queries = [
        "חוק הגנת הצרכן סעיף 2",
        "פסק דין חוזה אחיד תנאי מקפח",
    ]
    
    try:
        # Initialize retriever
        retriever = SearchResultsRetriever()
        
        print("Testing search results retrieval...")
        for i, query in enumerate(test_queries):
            print(f"\nQuery {i+1}: {query}")
            
            results = retriever.get_search_results(query, max_results=2)
            
            print(f"Found {len(results)} results")
            for j, result in enumerate(results):
                print(f"\nResult {j+1}: {result['title']}")
                print(f"URL: {result['link']}")
                print(f"Snippet: {result['snippet'][:100]}...")
                print(f"Full text length: {len(result['full_text'])} characters")
                print(f"Full text preview: {result['full_text'][:100]}...")
            
        print("\nSearch results retrieval testing complete!")
        
    except Exception as e:
        print(f"Error in test run: {e}")


if __name__ == "__main__":
    main()
