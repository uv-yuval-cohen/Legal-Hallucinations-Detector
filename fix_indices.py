import pandas as pd


def fix_csv_index(input_file, output_file=None, start_index=1):
    """
    Fix the index column in a CSV file by resetting it to sequential values.

    Parameters:
    input_file (str): Path to the input CSV file
    output_file (str): Path to the output CSV file (if None, overwrites input file)
    start_index (int): Starting value for the index (1 for 1-based, 0 for 0-based)
    """

    # Read the CSV file
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)

    # Display current index info
    print(f"Original data shape: {df.shape}")
    print(f"Current Index column range: {df['Index'].min()} to {df['Index'].max()}")

    # Fix the index column
    df['Index'] = range(start_index, len(df) + start_index)

    # Display fixed index info
    print(f"Fixed Index column range: {df['Index'].min()} to {df['Index'].max()}")

    # Save the file
    if output_file is None:
        output_file = input_file

    df.to_csv(output_file, index=False)
    print(f"Fixed CSV saved to: {output_file}")

    return df


# Example usage:
if __name__ == "__main__":
    # Fix the index column (1-based indexing from 1 to 519)
    fixed_df = fix_csv_index('annotated_paragraphs.csv', 'annotated_paragraphs_fixed.csv', start_index=1)

    # Alternative: Use 0-based indexing (0 to 518)
    # fixed_df = fix_csv_index('annotated_paragraphs.csv', 'annotated_paragraphs_fixed.csv', start_index=0)

    # Alternative: Overwrite the original file
    # fixed_df = fix_csv_index('annotated_paragraphs.csv', start_index=1)