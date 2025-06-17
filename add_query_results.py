import requests
import pandas as pd
from newspaper import Article
from bs4 import BeautifulSoup
import time
import re
import json
import os
from dotenv import load_dotenv
# PDF processing libraries
try:
    import PyPDF2

    PDF_SUPPORT = True
    print("PDF support enabled with PyPDF2")
except ImportError:
    try:
        import pdfplumber

        PDF_SUPPORT = True
        print("PDF support enabled with pdfplumber")
    except ImportError:
        PDF_SUPPORT = False
        print("Warning: No PDF support. Install PyPDF2 or pdfplumber for PDF extraction")


def extract_pdf_text(url, headers):
    """
    Extract text from PDF files.
    """
    if not PDF_SUPPORT:
        return ""

    try:
        response = requests.get(url, headers=headers, timeout=15)
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


def clean_text(text):
    """
    Clean extracted text from HTML artifacts and improve readability for Hebrew text.
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


def google_search(query, api_key, cx, target_results=5):
    """
    Perform a Google Custom Search API query and extract article text.
    Enhanced for Hebrew content with PDF support.
    """
    # Step 1: Perform the Google Search API query
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": api_key,
        "cx": cx,
        "num": 10,  # Get 10 results to have backup options
        "lr": "lang_he",  # Prefer Hebrew results
        "hl": "he"  # Hebrew interface language
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json().get("items", [])
    except requests.RequestException as e:
        print(f"Error during API request for query '{query}': {e}")
        return []

    if not results:
        return []

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

    # Step 3: Extract article text
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept-Language": "he,en;q=0.9",
        "Accept-Charset": "utf-8"
    }

    successful_results = []

    for result in all_results:
        if len(successful_results) >= target_results:
            break

        # Check if it's a PDF
        is_pdf = result["link"].lower().endswith('.pdf')

        if is_pdf:
            print(f"Processing PDF: {result['link']}")
            full_text = extract_pdf_text(result["link"], headers)
            if full_text and len(full_text.strip()) > 100:
                full_text = clean_text(full_text)
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
            article = Article(result["link"], headers=headers)
            article.download()
            article.parse()

            # Get the main article text
            full_text = clean_text(article.text)
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
                response = requests.get(result["link"], headers=headers, timeout=10)
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
                full_text = clean_text(full_text)

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


def create_bert_training_dataset(csv_file_path, api_key, cx,
                                 jsonl_output_file="bert_training_data.jsonl",
                                 progress_file="processing_progress.json",
                                 start_row=0, max_rows=None):
    """
    Process CSV and create JSONL training dataset for DictaBERT hallucination detection.
    ALWAYS redoes everything - no skipping.

    Args:
        csv_file_path: Path to the CSV file
        api_key: Google API key
        cx: Google Custom Search Engine ID
        jsonl_output_file: Path to JSONL output file for training
        progress_file: Path to save processing progress
        start_row: Row to start processing from
        max_rows: Maximum number of rows to process (None for all)
    """
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8')
    except UnicodeDecodeError:
        for encoding in ['utf-8-sig', 'cp1255', 'iso-8859-8']:
            try:
                df = pd.read_csv(csv_file_path, encoding=encoding)
                print(f"Successfully read CSV with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Could not read CSV file with any common encoding")

    print(f"CSV loaded successfully. Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Initialize progress tracking
    progress_data = {
        "total_rows": len(df),
        "processed_count": 0,
        "success_count": 0,
        "failed_count": 0,
        "no_query_count": 0,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # Load existing data to preserve other indices
    existing_data = {}
    if os.path.exists(jsonl_output_file):
        with open(jsonl_output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        existing_data[data['index']] = data
                    except:
                        pass
        print(f"📂 Loaded {len(existing_data)} existing records")

    print(f"🔄 ALWAYS REDOING - No skipping mode enabled")

    # Determine which rows to process
    end_row = min(start_row + max_rows, len(df)) if max_rows else len(df)

    for index, row in df.iloc[start_row:end_row].iterrows():
        search_query = row.get('search_query', '')
        row_index = row.get('Index', index)
        original_paragraph = row.get('paragraph', '')
        is_hallucination = int(row.get('isHallucination', 0))  # Convert to int for training

        print(f"\n{'=' * 80}")
        print(f"Processing Row {index} (Index: {row_index})")
        print(f"Label (isHallucination): {is_hallucination}")
        progress_data["processed_count"] += 1

        # Handle empty queries
        # Skip rows with no search query - BERT doesn't need them
        if pd.isna(search_query) or search_query.strip() == '':
            print(f"Row {index}: Skipping - no search query")
            progress_data["no_query_count"] += 1
            continue
        else:
            print(f"Original query: {search_query}")

            # Split multiple queries by '$!$'
            queries = [q.strip() for q in search_query.split('$!$') if q.strip()]
            print(f"Found {len(queries)} query(ies)")

            all_results_text = []

            for query_idx, query in enumerate(queries, 1):
                print(f"\n--- Processing Query {query_idx}/{len(queries)}: {query} ---")

                # Perform search
                results = google_search(query, api_key, cx, target_results=5)

                if not results:
                    print(f"No results found for query: {query}")
                    all_results_text.append("NO RESULTS")
                    continue

                print(f"Found {len(results)} successful results")

                # Collect text from all results for this query
                query_text_parts = []
                for result_idx, result in enumerate(results, 1):
                    print(f"Result {result_idx}: {result['title'][:100]}...")

                    if result['full_text']:
                        query_text_parts.append(result['full_text'])

                # Combine all text from this query
                if query_text_parts:
                    combined_query_text = "\n\n".join(query_text_parts)
                    all_results_text.append(combined_query_text)
                else:
                    all_results_text.append("NO USABLE TEXT EXTRACTED")

            # Create final search result
            if all_results_text:
                search_result = "\n\n".join(all_results_text)
                print(f"Final search result length: {len(search_result)} characters")
                progress_data["success_count"] += 1
            else:
                search_result = "NO RESULTS"
                progress_data["failed_count"] += 1

        # Create DictaBERT formatted text with clear markers
        # Format: [ORIGINAL] original_text [/ORIGINAL] [SEARCH] search_results [/SEARCH]
        bert_formatted_text = f"[ORIGINAL] {original_paragraph} [/ORIGINAL] [SEARCH] {search_result} [/SEARCH]"

        # Create training example
        training_example = {
            "index": int(row_index),
            "text": bert_formatted_text,
            "label": is_hallucination,
            "original_paragraph": original_paragraph,
            "search_query": search_query if not pd.isna(search_query) else "",
            "search_result_length": len(search_result)
        }

        # Update the data in memory
        existing_data[int(row_index)] = training_example

        print(f"✅ Training example created for Index {row_index}")
        print(f"Label: {is_hallucination}, Text length: {len(bert_formatted_text)} chars")

        # Save progress every 10 rows
        if progress_data["processed_count"] % 10 == 0:
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
            print(f"📊 Progress saved: {progress_data['processed_count']} processed")

        # Add delay between rows
        time.sleep(1)

    # Final progress save
    progress_data["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, ensure_ascii=False, indent=2)

    # Write all data back (existing + updated)
    with open(jsonl_output_file, 'w', encoding='utf-8') as f:
        for idx in sorted(existing_data.keys()):
            f.write(json.dumps(existing_data[idx], ensure_ascii=False) + '\n')
    print(f"💾 Saved {len(existing_data)} total records")

    print(f"\n{'=' * 80}")
    print(f"🎉 Processing completed successfully!")
    print(f"📊 Final Statistics:")
    print(f"   Total processed: {progress_data['processed_count']}")
    print(f"   Successful extractions: {progress_data['success_count']}")
    print(f"   Failed extractions: {progress_data['failed_count']}")
    print(f"   No query rows: {progress_data['no_query_count']}")
    print(f"📁 Training data saved to: {jsonl_output_file}")
    print(f"📈 Progress log saved to: {progress_file}")

    return jsonl_output_file, progress_file


def load_and_preview_training_data(jsonl_file, num_examples=3):
    """
    Load and preview the training data from JSONL file.
    """
    print(f"\n📖 Previewing training data from: {jsonl_file}")

    if not os.path.exists(jsonl_file):
        print(f"❌ File not found: {jsonl_file}")
        return

    examples = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num <= num_examples:
                try:
                    example = json.loads(line.strip())
                    examples.append(example)
                except json.JSONDecodeError as e:
                    print(f"❌ Error parsing line {line_num}: {e}")

    print(f"📊 Showing first {len(examples)} examples:")
    for i, example in enumerate(examples, 1):
        print(f"\n--- Example {i} ---")
        print(f"Index: {example['index']}")
        print(f"Label: {example['label']} ({'Hallucination' if example['label'] == 1 else 'Not Hallucination'})")
        print(f"Text preview: {example['text'][:200]}...")
        print(f"Full text length: {len(example['text'])} characters")




if __name__ == "__main__":

    # Load environment variables from .env file
    load_dotenv()
    
    # Configuration
    API_KEY = os.getenv('SEARCH_API_KEY')
    SEARCH_ENGINE_ID = "d01be1a5fa9ba41ab"
    CSV_FILE = "annotated_paragraphs.csv"

    # Output files
    JSONL_OUTPUT = "bert_training_data.jsonl"
    PROGRESS_FILE = "processing_progress.json"

    print("🚀 Starting DictaBERT Training Data Creation")
    print("Features enabled:")
    print(f"   - PDF extraction: {'✅ Yes' if PDF_SUPPORT else '❌ No (install PyPDF2 or pdfplumber)'}")
    print("   - No skipping: ✅ Always redo everything")
    print("   - Label inclusion: ✅ isHallucination column")
    print("   - DictaBERT markers: ✅ [ORIGINAL] and [SEARCH]")
    print("   - JSONL format: ✅ Ready for training")
    print("   - Progress tracking: ✅ Every 10 rows")

    # CHANGE max_rows=1 to max_rows=None for full processing
    try:
        jsonl_path, progress_path = create_bert_training_dataset(
            csv_file_path=CSV_FILE,
            api_key=API_KEY,
            cx=SEARCH_ENGINE_ID,
            jsonl_output_file=JSONL_OUTPUT,
            progress_file=PROGRESS_FILE,
            start_row=1,
            max_rows=None  # CHANGE TO None FOR FULL PROCESSING
        )

        # Preview the created training data
        load_and_preview_training_data(jsonl_path, num_examples=2)

        print(f"\n🎯 Ready for DictaBERT training!")
        print(f"📁 Training file: {jsonl_path}")
        print(f"📊 Progress file: {progress_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()