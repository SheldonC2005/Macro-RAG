# Attempt to import RecursiveCharacterTextSplitter from possible locations
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore  # noqa: F401
except ImportError:
    try:
        from langchain.text_splitters import RecursiveCharacterTextSplitter  # type: ignore  # noqa: F401
    except ImportError:
        # Simple fallback implementation
        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, *args, **kwargs):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap

            def split_text(self, text: str):
                chunks = []
                i = 0
                while i < len(text):
                    chunks.append(text[i : i + self.chunk_size])
                    i += self.chunk_size - self.chunk_overlap
                return chunks

from typing import List, Dict, Any
import uuid

from economic_graphrag.config import settings

def chunk_document(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Splits a document's content into smaller chunks using RecursiveCharacterTextSplitter.
    Each chunk inherits and extends the metadata of the parent document.

    Args:
        document (Dict[str, Any]): A dictionary representing a document with 'content' and metadata.

    Returns:
        List[Dict[str, Any]]: A list of chunk dictionaries.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    
    content = document.get("content", "")
    chunks_text = text_splitter.split_text(content)
    
    chunks = []
    for i, chunk_text in enumerate(chunks_text):
        chunk_id = f"{document.get('document_id', 'doc')}_chunk_{i}"
        chunk = {
            **document,  # Inherit metadata from parent document
            "chunk_id": chunk_id,
            "content": chunk_text,
        }
        # Ensure 'page' is present, default to 0 if not
        if 'page' not in chunk:
            chunk['page'] = 0
            
        chunks.append(chunk)
        
    return chunks

def chunk_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Takes a list of documents and returns a flattened list of all chunks.

    Args:
        documents (List[Dict[str, Any]]): A list of document dictionaries.

    Returns:
        List[Dict[str, Any]]: A list of all chunk dictionaries from all documents.
    """
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
    return all_chunks

if __name__ == '__main__':
    # Example Usage
    from economic_graphrag.ingestion.pdf_loader import load_pdf
    from economic_graphrag.config import settings

    # Use the dummy PDF from pdf_loader example
    dummy_pdf_path = settings.PDF_DIR / "dummy_report.pdf"
    
    if not dummy_pdf_path.exists():
        print("Dummy PDF not found. Please run pdf_loader.py first to create it.")
    else:
        # 1. Load documents
        loaded_docs = load_pdf(dummy_pdf_path)
        
        if loaded_docs:
            # 2. Chunk documents
            chunks = chunk_documents(loaded_docs)
            
            print(f"Split {len(loaded_docs)} document(s) into {len(chunks)} chunk(s).")
            
            if chunks:
                print("\n--- Example Chunk ---")
                print(chunks[0]["content"])
                print("\n--- Chunk Metadata ---")
                # Print metadata, but not the long content
                metadata = {k: v for k, v in chunks[0].items() if k != 'content'}
                print(metadata)
        else:
            print("No documents were loaded, so no chunking was performed.")
