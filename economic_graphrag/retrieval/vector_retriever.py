# economic_graphrag/retrieval/vector_retriever.py
from typing import List, Dict, Any, Tuple

from economic_graphrag.vector_store.chroma_manager import ChromaVectorStore

class VectorRetriever:
    """
    A class to handle retrieval of documents from a vector store.
    """

    def __init__(self, vector_store: ChromaVectorStore):
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5, filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Retrieves documents based on a query, with an optional metadata filter.

        Args:
            query (str): The query text.
            top_k (int): The number of top results to return.
            filter_metadata (Dict[str, Any], optional): Metadata to filter results. Defaults to None.

        Returns:
            List[Dict[str, Any]]: A list of retrieved chunk dictionaries, each including a 'score'.
        """
        if filter_metadata:
            results = self.vector_store.metadata_filtered_search(query, filter_metadata, top_k)
        else:
            results = self.vector_store.similarity_search(query, top_k)
        
        # Format results to include the score within the dictionary
        retrieved_chunks = []
        for chunk, score in results:
            chunk['score'] = score
            retrieved_chunks.append(chunk)
            
        return retrieved_chunks

if __name__ == '__main__':
    # Example Usage
    # This assumes that the ChromaDB has been populated by running previous examples.
    import shutil
    from pathlib import Path
    from economic_graphrag.config import settings
    from economic_graphrag.chunking.chunker import chunk_documents
    from economic_graphrag.ingestion.pdf_loader import load_pdf

    # Setup: Ensure there is data in ChromaDB
    db_path = Path(settings.CHROMA_PATH)
    if db_path.exists():
        shutil.rmtree(db_path)
    
    vector_store = ChromaVectorStore()
    dummy_pdf_path = settings.PDF_DIR / "dummy_report.pdf"

    if not dummy_pdf_path.exists():
        print("Dummy PDF not found. Please run pdf_loader.py first.")
    else:
        docs = load_pdf(dummy_pdf_path)
        chunks = chunk_documents(docs)
        if chunks:
            # Add a country to metadata for filtering example
            for chunk in chunks:
                chunk['country'] = 'Testland'
            vector_store.add_documents(chunks)

            # 1. Initialize the retriever
            retriever = VectorRetriever(vector_store)

            # 2. Perform a simple retrieval
            query = "What is the GDP of Testland?"
            print(f"\n--- Retrieving documents for query: '{query}' ---")
            retrieved_docs = retriever.retrieve(query, top_k=2)

            if retrieved_docs:
                print(f"Found {len(retrieved_docs)} documents.")
                for doc in retrieved_docs:
                    print(f"Score: {doc['score']:.4f}")
                    print(f"Content: {doc['content'][:100]}...") # Print snippet
                    print("-" * 10)
            else:
                print("No documents found.")

            # 3. Perform a retrieval with metadata filter
            filter_meta = {"country": "Testland"}
            print(f"\n--- Retrieving documents with filter: {filter_meta} ---")
            filtered_docs = retriever.retrieve(query, top_k=2, filter_metadata=filter_meta)

            if filtered_docs:
                print(f"Found {len(filtered_docs)} documents with filter.")
                for doc in filtered_docs:
                    print(f"Score: {doc['score']:.4f}")
                    print(f"Country: {doc.get('country')}")
                    print(f"Content: {doc['content'][:100]}...")
                    print("-" * 10)
            else:
                print("No documents found with the specified filter.")
        else:
            print("No chunks were created from the PDF.")
