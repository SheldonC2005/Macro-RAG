# economic_graphrag/ingestion/pdf_loader.py
import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import List, Dict, Any
import uuid

def load_pdf(file_path: Path) -> List[Dict[str, Any]]:
    """
    Loads a PDF file and extracts text and metadata from each page.
    It uses pdfplumber for robust text extraction and PyMuPDF for metadata.

    Args:
        file_path (Path): The path to the PDF file.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                               represents a page with its content and metadata.
    """
    documents = []
    try:
        # Use PyMuPDF to get metadata
        doc_pymupdf = fitz.open(file_path)
        metadata = doc_pymupdf.metadata
        title = metadata.get("title", file_path.stem)
        publication_date = metadata.get("creationDate", "")

        # Use pdfplumber for text extraction
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    documents.append({
                        "document_id": str(uuid.uuid4()),
                        "title": title,
                        "source": str(file_path),
                        "page": i + 1,
                        "content": text,
                        "publication_date": publication_date,
                        # These will be populated later if applicable
                        "country": None,
                        "indicator": None,
                    })
        doc_pymupdf.close()
    except Exception as e:
        print(f"Error loading PDF {file_path}: {e}")
    
    return documents

if __name__ == '__main__':
    # Example usage:
    # Create a dummy PDF for testing
    from economic_graphrag.config import settings
    
    dummy_pdf_path = settings.PDF_DIR / "dummy_report.pdf"
    if not settings.PDF_DIR.exists():
        settings.PDF_DIR.mkdir(parents=True)

    # Simple text file to simulate content
    if not dummy_pdf_path.exists():
        try:
            # Create a simple PDF with fitz
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 72), "This is a test report about the GDP of Testland.")
            page.insert_text((50, 92), "In 2023, the GDP growth was 5%.")
            doc.set_metadata({"title": "Dummy GDP Report", "author": "Test Author"})
            doc.save(dummy_pdf_path)
            doc.close()
            print(f"Created dummy PDF: {dummy_pdf_path}")
        except Exception as e:
            print(f"Could not create dummy PDF: {e}")


    if dummy_pdf_path.exists():
        loaded_docs = load_pdf(dummy_pdf_path)
        if loaded_docs:
            print(f"Successfully loaded {len(loaded_docs)} pages from {dummy_pdf_path.name}")
            print("\n--- First Page Content ---")
            print(loaded_docs[0]["content"])
            print("\n--- Metadata ---")
            print({k: v for k, v in loaded_docs[0].items() if k != 'content'})
        else:
            print(f"No documents loaded from {dummy_pdf_path.name}")
    else:
        print("Dummy PDF not found, skipping example.")
