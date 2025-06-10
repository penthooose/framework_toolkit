import os
import re
from pathlib import Path
from typing import List, Tuple
import PyPDF2
import pdfplumber
import argparse
import logging
import warnings

# Suppress warnings from pdfplumber about missing CropBox
warnings.filterwarnings("ignore", category=UserWarning, module="pdfplumber")
logging.getLogger("pdfplumber").setLevel(logging.ERROR)

MAINPATH = (
    r"C:\Users\Paul\OneDrive - Otto-Friedrich-Universität Bamberg\Masterarbeit\Research"
)


def extract_text_from_pdf(pdf_path: str) -> List[Tuple[str, dict]]:
    """
    Extract text from PDF with basic formatting information.
    Returns list of (text, format_info) tuples.
    """
    text_blocks = []

    try:
        # Temporarily suppress stdout to avoid CropBox warnings
        import sys
        from contextlib import redirect_stderr
        import io

        stderr_buffer = io.StringIO()
        with redirect_stderr(stderr_buffer):
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract text with some basic formatting detection
                    text = page.extract_text()
                    if text:
                        # Split into lines and analyze
                        lines = text.split("\n")
                        for line in lines:
                            line = line.strip()
                            if line:
                                # Basic format detection based on text characteristics
                                format_info = {
                                    "length": len(line),
                                    "is_upper": line.isupper(),
                                    "is_title": line.istitle(),
                                    "ends_with_punct": line.endswith(
                                        (".", "!", "?", ":")
                                    ),
                                    "page": page_num,
                                }
                                text_blocks.append((line, format_info))
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        # Fallback to PyPDF2
        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text:
                        lines = text.split("\n")
                        for line in lines:
                            line = line.strip()
                            if line:
                                format_info = {
                                    "length": len(line),
                                    "is_upper": line.isupper(),
                                    "is_title": line.istitle(),
                                    "ends_with_punct": line.endswith(
                                        (".", "!", "?", ":")
                                    ),
                                    "page": page_num,
                                }
                                text_blocks.append((line, format_info))
        except Exception as e2:
            print(f"Error with fallback method: {e2}")

    return text_blocks


def detect_headings(text_blocks: List[Tuple[str, dict]]) -> List[Tuple[str, int]]:
    """
    Detect headings based on text characteristics.
    Returns list of (text, heading_level) tuples where heading_level is 0 for normal text.
    """
    if not text_blocks:
        return []

    result = []

    for text, format_info in text_blocks:
        heading_level = 0

        # Heuristics for heading detection
        if len(text) < 100:  # Short lines are more likely to be headings
            if format_info["is_upper"]:
                heading_level = 1  # ALL CAPS = main heading
            elif format_info["is_title"]:
                heading_level = 2  # Title Case = subheading
            elif not format_info["ends_with_punct"] and len(text) < 50:
                heading_level = 3  # Short lines without punctuation

        # Check for numbered headings (1., 1.1, etc.)
        if re.match(r"^\d+\.(\d+\.)*\s", text):
            level_count = text.split()[0].count(".")
            heading_level = min(level_count, 6)

        # Check for common heading patterns
        if re.match(
            r"^(Chapter|Section|Part|Abstract|Introduction|Conclusion|References)",
            text,
            re.IGNORECASE,
        ):
            heading_level = max(heading_level, 2)

        result.append((text, heading_level))

    return result


def convert_to_markdown(text_blocks: List[Tuple[str, int]]) -> str:
    """
    Convert text blocks with heading information to Markdown format.
    """
    markdown_lines = []
    current_paragraph = []

    for text, heading_level in text_blocks:
        # Clean up text
        text = text.strip()
        if not text:
            continue

        if heading_level > 0:
            # Finish current paragraph if exists
            if current_paragraph:
                markdown_lines.append(" ".join(current_paragraph))
                current_paragraph = []
                markdown_lines.append("")  # Empty line after paragraph

            # Add heading
            heading_prefix = "#" * heading_level
            markdown_lines.append(f"{heading_prefix} {text}")
            markdown_lines.append("")  # Empty line after heading
        else:
            # Regular text - check if it should start a new paragraph
            if text.endswith((".", "!", "?", ":")):
                current_paragraph.append(text)
                markdown_lines.append(" ".join(current_paragraph))
                current_paragraph = []
                markdown_lines.append("")  # Empty line after paragraph
            else:
                current_paragraph.append(text)

    # Add remaining paragraph
    if current_paragraph:
        markdown_lines.append(" ".join(current_paragraph))

    return "\n".join(markdown_lines)


def convert_pdf_to_markdown(pdf_path: str, output_path: str) -> bool:
    """
    Convert a single PDF file to Markdown format.
    Returns True if successful, False otherwise.
    """
    try:
        print(f"Converting: {pdf_path}")

        # Extract text with formatting
        text_blocks = extract_text_from_pdf(pdf_path)

        if not text_blocks:
            print(f"Warning: No text found in {pdf_path}")
            return False

        # Detect headings
        text_with_headings = detect_headings(text_blocks)

        # Convert to Markdown
        markdown_content = convert_to_markdown(text_with_headings)

        # Write to output file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        print(f"Successfully converted to: {output_path}")
        return True

    except Exception as e:
        print(f"Error converting {pdf_path}: {str(e)}")
        return False


def process_folder_recursively(folder_path: str) -> None:
    """
    Recursively process all PDF files in the given folder and subfolders.
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    if not folder_path.is_dir():
        print(f"Error: '{folder_path}' is not a directory.")
        return

    pdf_files_found = 0
    converted_files = 0
    skipped_files = 0

    # Walk through all directories recursively
    for root, dirs, files in os.walk(folder_path):
        root_path = Path(root)

        # Process all PDF files in current directory
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files_found += 1
                pdf_path = root_path / file

                # Create output filename with .md extension
                md_filename = file[:-4] + ".md"  # Replace .pdf with .md
                md_path = root_path / md_filename

                # Check if markdown file already exists
                if md_path.exists():
                    print(f"Skipping: {pdf_path} (Markdown file already exists)")
                    skipped_files += 1
                    continue

                # Convert PDF to Markdown
                if convert_pdf_to_markdown(str(pdf_path), str(md_path)):
                    converted_files += 1

    print(f"\nProcessing complete!")
    print(f"PDF files found: {pdf_files_found}")
    print(f"Files successfully converted: {converted_files}")
    print(f"Files skipped (already exist): {skipped_files}")


def main():
    """
    Main function to handle command line arguments and start processing.
    """
    parser = argparse.ArgumentParser(
        description="Convert PDF files to Markdown format recursively"
    )
    parser.add_argument("folder_path", help="Path to the folder containing PDF files")

    args = parser.parse_args()

    print(f"Starting PDF to Markdown conversion...")
    print(f"Target folder: {args.folder_path}")
    print("-" * 50)

    process_folder_recursively(args.folder_path)


if __name__ == "__main__":
    # If no command line arguments, use MAINPATH as default
    import sys

    if len(sys.argv) == 1:
        folder_path = input(
            f"Enter the folder path containing PDF files (default: {MAINPATH}): "
        ).strip()
        if not folder_path:
            folder_path = MAINPATH

        print(f"Starting PDF to Markdown conversion...")
        print(f"Target folder: {folder_path}")
        print("-" * 50)
        process_folder_recursively(folder_path)
    else:
        main()
