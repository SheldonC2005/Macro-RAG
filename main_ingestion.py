# main_ingestion.py
import os
from pathlib import Path

from economic_graphrag.config import settings
from economic_graphrag.ingestion.pdf_loader import load_pdf
from economic_graphrag.ingestion.worldbank_loader import load_worldbank_data
from economic_graphrag.ingestion.fred_loader import load_fred_data
from economic_graphrag.ingestion.oecd_loader import load_oecd_data
from economic_graphrag.chunking.chunker import chunk_documents
from economic_graphrag.vector_store.chroma_manager import ChromaVectorStore
from economic_graphrag.graph.graph_builder import GraphBuilder
from economic_graphrag.graph.neo4j_manager import neo4j_manager

def run_ingestion_pipeline():
    """
    Orchestrates the entire data ingestion, processing, embedding, and storage pipeline.
    """
    print("--- Starting Data Ingestion Pipeline ---")

    # --- 0. Setup ---
    # Ensure data directories exist
    settings.PDF_DIR.mkdir(parents=True, exist_ok=True)
    settings.WORLDBANK_DIR.mkdir(parents=True, exist_ok=True)
    settings.FRED_DIR.mkdir(parents=True, exist_ok=True)
    settings.OECD_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Data from all sources ---
    all_docs = []
    print("\n[Phase 1/5] Loading data from sources...")

    # Load from APIs
    # Note: These make live API calls.
    all_docs.extend(load_worldbank_data())
    all_docs.extend(load_fred_data())
    all_docs.extend(load_oecd_data())


    # Load from PDFs
    pdf_files = list(settings.PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {settings.PDF_DIR}. Please add some PDF reports to run the full pipeline.")
    else:
        print(f"Found {len(pdf_files)} PDF(s). Loading...")
        for pdf_path in pdf_files:
            all_docs.extend(load_pdf(pdf_path))
    
    if not all_docs:
        print("No documents loaded. Exiting pipeline.")
        # Create a dummy doc to allow pipeline to run for demonstration
        print("Creating a dummy document to demonstrate the pipeline...")
        dummy_doc = {
            "document_id": "dummy-doc-1",
            "title": "Dummy Document",
            "source": "local",
            "content": "The GDP of Testland is expected to grow by 3% in the next fiscal year. Inflation remains a key factor.",
            "country": "Testland",
            "indicator": "GDP"
        }
        all_docs.append(dummy_doc)

    print(f"Total documents loaded: {len(all_docs)}")

    # --- 2. Chunk Documents ---
    print("\n[Phase 2/5] Chunking documents...")
    all_chunks = chunk_documents(all_docs)
    print(f"Total chunks created: {len(all_chunks)}")

    if not all_chunks:
        print("No chunks were created. Exiting.")
        return

    # --- 3. Store in Vector Database ---
    print("\n[Phase 3/5] Storing chunks in ChromaDB...")
    vector_store = ChromaVectorStore()
    # For a clean run, you might want to clear the collection, but for now we'll just add.
    vector_store.add_documents(all_chunks)
    print(f"ChromaDB collection '{settings.CHROMA_COLLECTION_NAME}' now has {vector_store.get_collection_size()} items.")

    # --- 4. Build Knowledge Graph ---
    print("\n[Phase 4/5] Building knowledge graph in Neo4j...")
    try:
        # Ensure constraints are set up in Neo4j
        neo4j_manager.create_constraints()
        
        graph_builder = GraphBuilder()
        # Note: This step can be slow as it involves LLM calls for entity extraction.
        # For a large number of chunks, you might process them in batches.
        graph_builder.build_graph_from_chunks(all_chunks)
        print("Knowledge graph build process initiated.")
    except Exception as e:
        print(f"Error building knowledge graph: {e}")
        print("Please ensure Neo4j and Ollama are running.")

    # --- 5. Cleanup ---
    print("\n[Phase 5/5] Cleaning up connections...")
    neo4j_manager.close()
    print("Neo4j connection closed.")

    print("\n--- Data Ingestion Pipeline Complete ---")

if __name__ == "__main__":
    # This script is the main entry point for populating the RAG system.
    run_ingestion_pipeline()
