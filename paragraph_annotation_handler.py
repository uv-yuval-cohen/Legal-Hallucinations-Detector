"""
Module: paragraph_annotation_handler.py

Description:
This module handles text processing and annotation management for the Legal Hallucination Detector pipeline. 
It provides functionality for paragraph extraction, dataset management, and search query handling.

Key Components:
1. Text Processing
   - Splits legal text into paragraphs while preserving language-specific features
   - Supports multilingual content including Hebrew

2. Dataset Management
   - Maintains a CSV-based annotation system for legal paragraphs
   - Tracks metadata including need_check, isHallucination flags, and search queries
   - Provides functions for adding, deleting, and updating paragraph records
   - Prevents duplicate entries through automatic detection

3. Search Query Handling
   - Adds and manages search queries for paragraphs requiring verification
   - Supports batch processing from external files (for_search.txt)

4. Statistics and Reporting
   - Generates dataset composition statistics
   - Tracks annotation progress and verification status

5. Interactive Interface
   - Command-line menu system for manual data curation
   - Batch processing capabilities for efficient data handling

This module forms the foundation of the data preparation phase in the hallucination 
detection pipeline, creating properly annotated datasets for the classification models.
"""

import csv
import os


def split_text_to_paragraphs(text):
    """
    Splits text into paragraphs.
    Works with text in any language, including Hebrew.

    Parameters:
    text (str): The text to split into paragraphs

    Returns:
    list: A list of paragraphs
    """
    # Split the text by new lines
    lines = text.splitlines()

    paragraphs = []
    current_paragraph = []

    for line in lines:
        # If the line is empty and we already have a paragraph in progress,
        # add the paragraph to our list of paragraphs
        if not line.strip() and current_paragraph:
            paragraphs.append('\n'.join(current_paragraph))
            current_paragraph = []
        # If the line isn't empty, add it to the current paragraph
        elif line.strip():
            current_paragraph.append(line)

    # If there's a remaining paragraph, add it too
    if current_paragraph:
        paragraphs.append('\n'.join(current_paragraph))

    return paragraphs


def add_paragraph_to_csv(paragraph, need_check, is_hallucination, search_query="", csv_file="annotated_paragraphs.csv"):
    """
    Adds a single paragraph with tags to a CSV file including an index column

    Parameters:
    paragraph (str): The paragraph to add
    need_check (bool): Whether the paragraph needs checking
    is_hallucination (bool): Whether the paragraph contains a hallucination
    search_query (str): The search query associated with this paragraph
    csv_file (str): The CSV filename (default: annotated_paragraphs.csv)

    Returns:
    bool: Whether the addition was successful
    """
    # Check if the file exists
    file_exists = os.path.isfile(csv_file)

    try:
        # Determine the appropriate index
        index = 1  # Default starting value

        if file_exists:
            # Read the file to determine the next index
            with open(csv_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)

                if len(rows) > 1:  # If there are more rows than just the header
                    # Find the highest index and add 1
                    max_index = 0
                    for row in rows[1:]:  # Skip header row
                        if len(row) > 0 and row[0].isdigit():
                            max_index = max(max_index, int(row[0]))
                    index = max_index + 1

        # Open the file in append mode
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Create headers if the file is new
            if not file_exists:
                writer.writerow(['index', 'paragraph', 'need_check', 'isHallucination', 'search_query'])

            # Add the data with the index, converting booleans to 1/0 for machine learning compatibility
            writer.writerow([index, paragraph, 1 if need_check else 0, 1 if is_hallucination else 0, search_query])

        print(f"Paragraph added successfully to {csv_file} with index {index}")
        return True
    except Exception as e:
        print(f"Error adding paragraph to CSV: {e}")
        return False


def delete_row_by_index(target_index, csv_file="annotated_paragraphs.csv"):
    """
    Deletes a row by its index value in the first column

    Parameters:
    target_index (int): The index value of the row to delete
    csv_file (str): The CSV filename (default: annotated_paragraphs.csv)

    Returns:
    bool: Whether the deletion was successful
    """
    # Check if the file exists
    if not os.path.isfile(csv_file):
        print(f"The file {csv_file} does not exist")
        return False

    # Read all data from the file
    try:
        rows = []
        row_found = False
        deleted_row = None

        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Save the header row
            rows = [header]  # Add the header to the new rows list

            # Scan all rows
            for row in reader:
                if len(row) > 0 and row[0].isdigit() and int(row[0]) == target_index:
                    deleted_row = row  # Save the deleted row
                    row_found = True
                else:
                    rows.append(row)  # Add the row to the new list

        if not row_found:
            print(f"No row found with index {target_index}")
            return False

        # Write the data back to the file
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        print(f"Row with index {target_index} deleted successfully:")
        print(f"Content of deleted row: {deleted_row}")
        return True

    except Exception as e:
        print(f"Error deleting row from CSV: {e}")
        return False


def read_csv_data(csv_file="annotated_paragraphs.csv"):
    """
    Reads data from the CSV file

    Parameters:
    csv_file (str): The CSV filename (default: annotated_paragraphs.csv)

    Returns:
    list: A list of all records in the file
    """
    if not os.path.isfile(csv_file):
        print(f"The file {csv_file} does not exist")
        return []

    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []


def display_data_statistics():
    """
    Display statistics about the data in the CSV file
    """
    try:
        # Check if file exists
        if not os.path.isfile("annotated_paragraphs.csv"):
            print("No data file found. Statistics unavailable.")
            return

        # Read all data from the CSV file
        with open("annotated_paragraphs.csv", 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header row
            rows = list(reader)

        # Calculate statistics
        total_lines = len(rows)

        if total_lines == 0:
            print("The data file is empty. No statistics available.")
            return

        need_check_count = 0
        hallucination_count = 0
        search_query_count = 0
        search_query_check_count = 0  # Count of entries with search queries that need checking

        for row in rows:
            # Check if the need_check column is 1 (converted from boolean)
            needs_checking = len(row) >= 3 and row[2] == '1'
            has_search_query = len(row) >= 5 and row[4].strip()

            if needs_checking:
                need_check_count += 1
                # Check if isHallucination column is 1 (converted from boolean)
                if len(row) >= 4 and row[3] == '1':
                    hallucination_count += 1
                # Count search queries for entries that need checking
                if has_search_query:
                    search_query_check_count += 1

            # Count all search queries
            if has_search_query:
                search_query_count += 1

        # Calculate percentages
        no_check_count = total_lines - need_check_count
        no_hallucination_count = need_check_count - hallucination_count

        need_check_percent = (need_check_count / total_lines) * 100 if total_lines > 0 else 0
        hallucination_percent = (hallucination_count / need_check_count) * 100 if need_check_count > 0 else 0

        search_query_percent = (search_query_count / total_lines) * 100 if total_lines > 0 else 0
        search_query_check_percent = (search_query_check_count / need_check_count) * 100 if need_check_count > 0 else 0

        # Display the statistics
        print("\n" + "=" * 50)
        print("DATA STATISTICS")
        print("=" * 50)
        print(f"Total paragraphs in database: {total_lines}")
        print("\nNeed checking:")
        print(f"  - YES: {need_check_count} / {total_lines} ({need_check_percent:.1f}%)")
        print(f"  - NO:  {no_check_count} / {total_lines} ({100 - need_check_percent:.1f}%)")

        if need_check_count > 0:
            print("\nFrom paragraphs needing check, contain hallucinations:")
            print(f"  - YES: {hallucination_count} / {need_check_count} ({hallucination_percent:.1f}%)")
            print(f"  - NO:  {no_hallucination_count} / {need_check_count} ({100 - hallucination_percent:.1f}%)")

            print("\nSearch queries for paragraphs needing check:")
            print(
                f"  - Have queries: {search_query_check_count} / {need_check_count} ({search_query_check_percent:.1f}%)")
            print(
                f"  - Missing queries: {need_check_count - search_query_check_count} / {need_check_count} ({100 - search_query_check_percent:.1f}%)")
        else:
            print("\nNo paragraphs marked as needing checking.")

        print(
            f"\nTotal paragraphs with search queries: {search_query_count} / {total_lines} ({search_query_percent:.1f}%)")

    except Exception as e:
        print(f"Error retrieving statistics: {e}")


def add_data_from_file(file_path=None):
    """
    Process text from a file, split it into paragraphs,
    and add them to CSV with user-provided annotations

    Parameters:
    file_path (str, optional): Path to the text file. If None, it will be requested from the user.
    """
    if file_path is None:
        # Ask for the file path
        file_path = input("Enter the path to the text file: ")

    try:
        # Read the text file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Split into paragraphs
        paragraphs = split_text_to_paragraphs(text)

        print(f"Found {len(paragraphs)} paragraphs in the file.")

        # Get existing paragraphs from CSV to check for duplicates
        existing_paragraphs = set()
        try:
            if os.path.exists("annotated_paragraphs.csv"):
                with open("annotated_paragraphs.csv", 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    for row in reader:
                        if len(row) > 1:  # Make sure row has enough columns
                            existing_paragraphs.add(row[1])  # Add paragraph text to set
        except Exception as e:
            print(f"Warning: Could not read existing paragraphs: {e}")

        # Process each paragraph
        skipped = 0
        for i, paragraph in enumerate(paragraphs, 1):
            # Check if this paragraph already exists in the CSV
            if paragraph in existing_paragraphs:
                skipped += 1
                continue

            print(f"\nParagraph {i}/{len(paragraphs)}:")
            print("-" * 40)
            print(paragraph)
            print("-" * 40)

            # Ask if the paragraph needs checking
            while True:
                need_check_input = input("Does this paragraph need checking? (yes/no): ").strip().lower()
                if need_check_input in ["yes", "y"]:
                    need_check = True
                    break
                elif need_check_input in ["no", "n"]:
                    need_check = False
                    break
                else:
                    print("Invalid input. Please answer 'yes' or 'no'.")

            # If it needs checking, ask if it contains hallucinations
            is_hallucination = False
            if need_check:
                while True:
                    hallucination_input = input(
                        "Does this paragraph contain hallucinations? (yes/no): ").strip().lower()
                    if hallucination_input in ["yes", "y"]:
                        is_hallucination = True
                        break
                    elif hallucination_input in ["no", "n"]:
                        is_hallucination = False
                        break
                    else:
                        print("Invalid input. Please answer 'yes' or 'no'.")

            # Ask for a search query
            search_query = input("Enter a search query for this paragraph (press Enter to skip): ").strip()

            # Display a summary of the user's choices
            print("\nSummary of your selections:")
            print(f"- Needs checking: {'Yes' if need_check else 'No'}")
            if need_check:
                print(f"- Contains hallucinations: {'Yes' if is_hallucination else 'No'}")
            print(f"- Search query: {search_query if search_query else 'Not provided'}")

            # Add to CSV
            add_paragraph_to_csv(paragraph, need_check, is_hallucination, search_query)

            # Add to our local set to avoid duplicates in the same run
            existing_paragraphs.add(paragraph)

            print("Paragraph added to CSV file.")

            # Ask if the user wants to continue or stop
            if i < len(paragraphs):
                while True:
                    continue_input = input("\nContinue to the next paragraph? (yes/no): ").strip().lower()
                    if continue_input in ["yes", "y"]:
                        break
                    elif continue_input in ["no", "n"]:
                        print("Stopping the process. The processed paragraphs have been saved.")
                        return
                    else:
                        print("Invalid input. Please answer 'yes' or 'no'.")

        if skipped > 0:
            print(f"\n{skipped} paragraph(s) were already in the database and were skipped.")

        processed = len(paragraphs) - skipped
        print(f"\n{processed} paragraph(s) have been processed and added to the CSV file.")

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def delete_data():
    """
    Delete a row from the CSV file by its index
    """
    # Display the current data
    data = read_csv_data()

    if not data:
        print("No data available in the CSV file.")
        return

    print("\nCurrent data in the CSV file:")
    print("-" * 40)
    for row in data:
        print(f"Index: {row['index']}")
        print(f"Paragraph: {row['paragraph'][:50]}..." if len(
            row['paragraph']) > 50 else f"Paragraph: {row['paragraph']}")
        print(f"Needs checking: {row['need_check']}")
        print(f"Contains hallucinations: {row['isHallucination']}")
        search_query = row.get('search_query', '')
        print(f"Search query: {search_query if search_query else 'Not provided'}")
        print("-" * 40)

    # Ask for the index to delete
    try:
        index_to_delete = int(input("Enter the index of the row to delete: "))

        # Confirm deletion
        confirm = input(f"Are you sure you want to delete row with index {index_to_delete}? (yes/no): ").strip().lower()

        if confirm in ["yes", "y"]:
            if delete_row_by_index(index_to_delete):
                print(f"Row with index {index_to_delete} has been deleted.")
            else:
                print("Deletion failed.")
        else:
            print("Deletion cancelled.")

    except ValueError:
        print("Invalid input. Please enter a valid number.")
    except Exception as e:
        print(f"An error occurred: {e}")


def add_search_queries():
    """
    Add search queries to existing entries in the CSV file.
    Automatically skips entries with need_check=0.
    """
    # Check if file exists
    if not os.path.isfile("annotated_paragraphs.csv"):
        print("No data file found. Cannot add search queries.")
        return

    try:
        # Read all data from the CSV file
        rows = []
        indices = []
        need_check_indices = []  # Only indices that need checking
        with open("annotated_paragraphs.csv", 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Save the header
            rows = [header]  # Start with the header

            # If the header doesn't have a search_query column, add it
            if len(header) < 5 or header[4] != 'search_query':
                header.append('search_query')
                rows[0] = header

            # Read all rows and collect indices
            for row in reader:
                rows.append(row)
                if row and row[0].isdigit():
                    index = int(row[0])
                    indices.append(index)

                    # Only collect indices with need_check=1
                    if len(row) >= 3 and row[2] == '1':
                        need_check_indices.append(index)

                    # Ensure the row has enough columns
                    while len(row) < 5:
                        row.append("")

        if not indices:
            print("No data found in the CSV file.")
            return

        if not need_check_indices:
            print("No entries found that need checking. No search queries needed.")
            return

        min_index = min(need_check_indices)
        max_index = max(need_check_indices)

        print(f"\nAvailable index range for entries that need checking: {min_index} to {max_index}")
        print(f"Total entries that need checking: {len(need_check_indices)}")

        # Ask for starting index
        try:
            start_index = int(input(f"Enter the index to start from ({min_index}-{max_index}): "))
            if start_index < min_index or start_index > max_index:
                print(f"Index must be between {min_index} and {max_index}. Using {min_index}.")
                start_index = min_index
        except ValueError:
            print(f"Invalid input. Using the minimum index {min_index}.")
            start_index = min_index

        # Process rows starting from start_index
        print("\nStarting search query addition process...")
        modified_count = 0
        skipped_count = 0

        # Find the next valid index to start from (that needs checking)
        valid_start = False
        for i, index in enumerate(sorted(need_check_indices)):
            if index >= start_index:
                start_index = index
                valid_start = True
                break

        if not valid_start:
            print("No entries requiring checking found from the specified index.")
            return

        for i, row in enumerate(rows[1:], 1):  # Skip header
            if not row or not row[0].isdigit():  # Skip invalid rows
                continue

            index = int(row[0])
            if index < start_index:  # Skip rows before start_index
                continue

            # Skip rows that don't need checking
            if len(row) >= 3 and row[2] != '1':
                skipped_count += 1
                print(f"\nSkipping entry with index {index} (doesn't need checking)")
                continue

            print(f"\nProcessing entry with index {index}:")
            print("-" * 40)
            print(f"Paragraph: {row[1]}")
            print("-" * 40)

            # Show current search query if it exists
            current_query = row[4] if len(row) > 4 and row[4] else "None"
            print(f"Current search query: {current_query}")

            # Ask for new search query
            new_query = input("Enter a search query for this paragraph (press Enter to skip): ").strip()

            if new_query:
                # Update the search query
                row[4] = new_query
                modified_count += 1
                print(f"Search query updated to: {new_query}")
            else:
                print("No change made to this entry.")

            # Ask if the user wants to continue or stop
            if i < len(rows) - 1:  # If not the last row
                while True:
                    continue_input = input("\nContinue to the next paragraph? (yes/no): ").strip().lower()
                    if continue_input in ["yes", "y"]:
                        break
                    elif continue_input in ["no", "n"]:
                        print("Stopping the process. The changes have been saved.")
                        break
                    else:
                        print("Invalid input. Please answer 'yes' or 'no'.")

                if continue_input in ["no", "n"]:
                    break

        # Write modified data back to CSV
        with open("annotated_paragraphs.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        print(f"\n{modified_count} search queries have been added or updated.")
        print(f"{skipped_count} entries were skipped (didn't need checking).")

    except Exception as e:
        print(f"Error adding search queries: {e}")


def add_whole_file():
    """
    Add whole file with predefined settings via submenu
    """
    while True:
        print("\n" + "=" * 50)
        print("ADD WHOLE FILE SUBMENU")
        print("=" * 50)
        print("1. Add needs check and hallucination")
        print("2. Add doesn't need check (and no hallucination)")
        print("3. Add needs check but no hallucination")
        print("4. Add queries from file")
        print("5. Back to main menu")
        print("-" * 50)

        choice = input("Enter your choice (1-5): ").strip()

        if choice == "1":
            # Add entire file with need_check=1 and is_hallucination=1
            add_file_with_settings(need_check=True, is_hallucination=True)
        elif choice == "2":
            # Add entire file with need_check=0 and is_hallucination=0
            add_file_with_settings(need_check=False, is_hallucination=False)
        elif choice == "3":
            # Add entire file with need_check=1 and is_hallucination=0
            add_file_with_settings(need_check=True, is_hallucination=False)
        elif choice == "4":
            # Add search queries from file
            add_search_queries_from_file()
        elif choice == "5":
            print("Returning to main menu...")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")


def add_file_with_settings(need_check, is_hallucination):
    """
    Add all paragraphs from LLM_output.txt with specified settings

    Parameters:
    need_check (bool): Whether the paragraphs need checking
    is_hallucination (bool): Whether the paragraphs contain hallucinations
    """
    file_path = "LLM_output.txt"

    try:
        # Check if file exists
        if not os.path.isfile(file_path):
            print(f"Error: The file {file_path} was not found.")
            return

        # Read the text file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Split into paragraphs
        paragraphs = split_text_to_paragraphs(text)

        print(f"Found {len(paragraphs)} paragraphs in the file.")

        # Get existing paragraphs from CSV to check for duplicates
        existing_paragraphs = set()
        try:
            if os.path.exists("annotated_paragraphs.csv"):
                with open("annotated_paragraphs.csv", 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    for row in reader:
                        if len(row) > 1:  # Make sure row has enough columns
                            existing_paragraphs.add(row[1])  # Add paragraph text to set
        except Exception as e:
            print(f"Warning: Could not read existing paragraphs: {e}")

        # Process each paragraph
        added_count = 0
        skipped_count = 0

        for paragraph in paragraphs:
            # Check if this paragraph already exists in the CSV
            if paragraph in existing_paragraphs:
                skipped_count += 1
                continue

            # Add to CSV with specified settings
            if add_paragraph_to_csv(paragraph, need_check, is_hallucination):
                added_count += 1

            # Add to our local set to avoid duplicates in the same run
            existing_paragraphs.add(paragraph)

        # Show summary
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Total paragraphs in file: {len(paragraphs)}")
        print(f"Paragraphs added: {added_count}")
        print(f"Paragraphs skipped (duplicates): {skipped_count}")
        print(f"\nSettings applied:")
        print(f"- Needs checking: {'Yes' if need_check else 'No'}")
        print(f"- Contains hallucinations: {'Yes' if is_hallucination else 'No'}")

    except Exception as e:
        print(f"An error occurred: {e}")


def add_search_queries_from_file():
    """
    Add search queries from for_search.txt file to existing entries
    """
    search_file = "for_search.txt"

    # Check if search file exists
    if not os.path.isfile(search_file):
        print(f"Error: The file {search_file} was not found.")
        return

    # Check if data file exists
    if not os.path.isfile("annotated_paragraphs.csv"):
        print("No data file found. Cannot add search queries.")
        return

    try:
        # Read search queries from file
        with open(search_file, 'r', encoding='utf-8') as f:
            search_queries = [line.strip() for line in f if line.strip()]

        if not search_queries:
            print(f"No search queries found in {search_file}.")
            return

        print(f"Found {len(search_queries)} search queries in the file.")

        # Read CSV data
        rows = []
        indices = []
        need_check_indices = []  # Only indices that need checking

        with open("annotated_paragraphs.csv", 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Save the header
            rows = [header]  # Start with the header

            # If the header doesn't have a search_query column, add it
            if len(header) < 5 or header[4] != 'search_query':
                header.append('search_query')
                rows[0] = header

            # Read all rows and collect indices
            for row in reader:
                rows.append(row)
                if row and row[0].isdigit():
                    index = int(row[0])
                    indices.append(index)

                    # Only collect indices with need_check=1
                    if len(row) >= 3 and row[2] == '1':
                        need_check_indices.append(index)

                    # Ensure the row has enough columns
                    while len(row) < 5:
                        row.append("")

        if not indices:
            print("No data found in the CSV file.")
            return

        if not need_check_indices:
            print("No entries found that need checking. No search queries needed.")
            return

        # Sort indices for sequential processing
        need_check_indices.sort()
        min_index = min(need_check_indices)
        max_index = max(need_check_indices)

        print(f"\nAvailable index range for entries that need checking: {min_index} to {max_index}")
        print(f"Total entries that need checking: {len(need_check_indices)}")
        print(f"Total search queries to add: {len(search_queries)}")

        # Ask for starting index
        try:
            start_index = int(input(f"Enter the index to start from ({min_index}-{max_index}): "))
            if start_index < min_index or start_index > max_index:
                print(f"Index must be between {min_index} and {max_index}. Using {min_index}.")
                start_index = min_index
        except ValueError:
            print(f"Invalid input. Using the minimum index {min_index}.")
            start_index = min_index

        # Find the starting position in the need_check_indices list
        start_pos = 0
        for i, idx in enumerate(need_check_indices):
            if idx >= start_index:
                start_pos = i
                break

        # Check if we have enough entries for all search queries
        remaining_entries = len(need_check_indices) - start_pos
        if remaining_entries < len(search_queries):
            print(
                f"Warning: You have {len(search_queries)} search queries but only {remaining_entries} entries that need checking from index {start_index}.")
            print("Some search queries will not be used.")

        # Process rows and add search queries
        added_count = 0
        query_index = 0

        for index in need_check_indices[start_pos:]:
            # Break if we've used all search queries
            if query_index >= len(search_queries):
                break

            # Find the row with this index
            for row in rows[1:]:  # Skip header
                if row and row[0].isdigit() and int(row[0]) == index:
                    # Add the search query
                    row[4] = search_queries[query_index]
                    added_count += 1
                    query_index += 1
                    print(f"Added search query to index {index}: {search_queries[query_index - 1]}")
                    break

        # Write modified data back to CSV
        with open("annotated_paragraphs.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        print(f"\n{added_count} search queries have been added from the file.")

        # Report on unused queries
        unused_queries = len(search_queries) - query_index
        if unused_queries > 0:
            print(f"{unused_queries} search queries were not used (not enough entries).")

    except Exception as e:
        print(f"Error adding search queries from file: {e}")


def add_file_with_settings(need_check, is_hallucination):
    """
    Add all paragraphs from LLM_output.txt with specified settings

    Parameters:
    need_check (bool): Whether the paragraphs need checking
    is_hallucination (bool): Whether the paragraphs contain hallucinations
    """
    file_path = "LLM_output.txt"

    try:
        # Check if file exists
        if not os.path.isfile(file_path):
            print(f"Error: The file {file_path} was not found.")
            return

        # Read the text file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Split into paragraphs
        paragraphs = split_text_to_paragraphs(text)

        print(f"Found {len(paragraphs)} paragraphs in the file.")

        # Get existing paragraphs from CSV to check for duplicates
        existing_paragraphs = set()
        try:
            if os.path.exists("annotated_paragraphs.csv"):
                with open("annotated_paragraphs.csv", 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    for row in reader:
                        if len(row) > 1:  # Make sure row has enough columns
                            existing_paragraphs.add(row[1])  # Add paragraph text to set
        except Exception as e:
            print(f"Warning: Could not read existing paragraphs: {e}")

        # Process each paragraph
        added_count = 0
        skipped_count = 0

        for paragraph in paragraphs:
            # Check if this paragraph already exists in the CSV
            if paragraph in existing_paragraphs:
                skipped_count += 1
                continue

            # Add to CSV with specified settings
            if add_paragraph_to_csv(paragraph, need_check, is_hallucination):
                added_count += 1

            # Add to our local set to avoid duplicates in the same run
            existing_paragraphs.add(paragraph)

        # Show summary
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Total paragraphs in file: {len(paragraphs)}")
        print(f"Paragraphs added: {added_count}")
        print(f"Paragraphs skipped (duplicates): {skipped_count}")
        print(f"\nSettings applied:")
        print(f"- Needs checking: {'Yes' if need_check else 'No'}")
        print(f"- Contains hallucinations: {'Yes' if is_hallucination else 'No'}")

    except Exception as e:
        print(f"An error occurred: {e}")


def add_search_queries_from_file():
    """
    Add search queries from for_search.txt file to existing entries
    """
    search_file = "for_search.txt"

    # Check if search file exists
    if not os.path.isfile(search_file):
        print(f"Error: The file {search_file} was not found.")
        return

    # Check if data file exists
    if not os.path.isfile("annotated_paragraphs.csv"):
        print("No data file found. Cannot add search queries.")
        return

    try:
        # Read search queries from file
        with open(search_file, 'r', encoding='utf-8') as f:
            search_queries = [line.strip() for line in f if line.strip()]

        if not search_queries:
            print(f"No search queries found in {search_file}.")
            return

        print(f"Found {len(search_queries)} search queries in the file.")

        # Read CSV data
        rows = []
        indices = []
        need_check_indices = []  # Only indices that need checking

        with open("annotated_paragraphs.csv", 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Save the header
            rows = [header]  # Start with the header

            # If the header doesn't have a search_query column, add it
            if len(header) < 5 or header[4] != 'search_query':
                header.append('search_query')
                rows[0] = header

            # Read all rows and collect indices
            for row in reader:
                rows.append(row)
                if row and row[0].isdigit():
                    index = int(row[0])
                    indices.append(index)

                    # Only collect indices with need_check=1
                    if len(row) >= 3 and row[2] == '1':
                        need_check_indices.append(index)

                    # Ensure the row has enough columns
                    while len(row) < 5:
                        row.append("")

        if not indices:
            print("No data found in the CSV file.")
            return

        if not need_check_indices:
            print("No entries found that need checking. No search queries needed.")
            return

        # Sort indices for sequential processing
        need_check_indices.sort()
        min_index = min(need_check_indices)
        max_index = max(need_check_indices)

        print(f"\nAvailable index range for entries that need checking: {min_index} to {max_index}")
        print(f"Total entries that need checking: {len(need_check_indices)}")
        print(f"Total search queries to add: {len(search_queries)}")

        # Ask for starting index
        try:
            start_index = int(input(f"Enter the index to start from ({min_index}-{max_index}): "))
            if start_index < min_index or start_index > max_index:
                print(f"Index must be between {min_index} and {max_index}. Using {min_index}.")
                start_index = min_index
        except ValueError:
            print(f"Invalid input. Using the minimum index {min_index}.")
            start_index = min_index

        # Find the starting position in the need_check_indices list
        start_pos = 0
        for i, idx in enumerate(need_check_indices):
            if idx >= start_index:
                start_pos = i
                break

        # Check if we have enough entries for all search queries
        remaining_entries = len(need_check_indices) - start_pos
        if remaining_entries < len(search_queries):
            print(
                f"Warning: You have {len(search_queries)} search queries but only {remaining_entries} entries that need checking from index {start_index}.")
            print("Some search queries will not be used.")

        # Process rows and add search queries
        added_count = 0
        query_index = 0

        for index in need_check_indices[start_pos:]:
            # Break if we've used all search queries
            if query_index >= len(search_queries):
                break

            # Find the row with this index
            for row in rows[1:]:  # Skip header
                if row and row[0].isdigit() and int(row[0]) == index:
                    # Add the search query
                    row[4] = search_queries[query_index]
                    added_count += 1
                    query_index += 1
                    print(f"Added search query to index {index}: {search_queries[query_index - 1]}")
                    break

        # Write modified data back to CSV
        with open("annotated_paragraphs.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        print(f"\n{added_count} search queries have been added from the file.")

        # Report on unused queries
        unused_queries = len(search_queries) - query_index
        if unused_queries > 0:
            print(f"{unused_queries} search queries were not used (not enough entries).")

    except Exception as e:
        print(f"Error adding search queries from file: {e}")


def update_csv_structure():
    """
    Update the CSV file structure to include the search_query column if it doesn't exist
    """
    csv_file = "annotated_paragraphs.csv"
    if not os.path.isfile(csv_file):
        print(f"The file {csv_file} does not exist. No update needed.")
        return

    try:
        # Read the entire CSV file
        rows = []
        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)

            # Check if search_query column already exists
            if len(header) >= 5 and header[4] == 'search_query':
                print("CSV structure is already up to date.")
                return

            # Add search_query to header
            header.append('search_query')
            rows.append(header)

            # Add empty search_query to all rows
            for row in reader:
                row.append('')
                rows.append(row)

        # Write back to the CSV file
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        print("CSV structure updated successfully. Added 'search_query' column.")

    except Exception as e:
        print(f"Error updating CSV structure: {e}")


def main_menu():
    """
    Display the main menu and handle user choices
    """
    # Check and update CSV structure if needed
    if os.path.isfile("annotated_paragraphs.csv"):
        update_csv_structure()

    while True:
        print("\n" + "=" * 50)
        print("PARAGRAPH ANNOTATION TOOL")
        print("=" * 50)
        print("1. Add data from LLM_output.txt (default file)")
        print("2. Add data from a custom file path")
        print("3. Delete a row from data")
        print("4. Display data statistics")
        print("5. Add search queries to existing entries")
        print("6. Add whole file (batch processing)")
        print("7. Exit")
        print("-" * 50)

        choice = input("Enter your choice (1-7): ").strip()

        if choice == "1":
            # Use default file path
            default_path = "LLM_output.txt"
            print(f"Using default file: {default_path}")
            add_data_from_file(default_path)
        elif choice == "2":
            # Custom file path
            add_data_from_file()
        elif choice == "3":
            delete_data()
        elif choice == "4":
            display_data_statistics()
        elif choice == "5":
            add_search_queries()
        elif choice == "6":
            add_whole_file()
        elif choice == "7":
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 7.")


if __name__ == "__main__":
    main_menu()
