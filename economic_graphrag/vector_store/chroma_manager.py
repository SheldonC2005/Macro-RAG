# economic_graphrag/vector_store/chroma_manager.py
import chromadb
from typing import List, Dict, Any, Tuple

from economic_graphrag.config import settings
from economic_graphrag.embeddings.embedder import embedder
from economic_graphrag.vector_store.base import BaseVectorStore

class ChromaVectorStore(BaseVectorStore):
    """
    Implementation of BaseVectorStore using ChromaDB.
    """

    def __init__(self, path: str = settings.CHROMA_PATH, collection_name: str = settings.CHROMA_COLLECTION_NAME):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """
        Adds documents (chunks) to the ChromaDB collection.

        Args:
            chunks (List[Dict[str, Any]]): A list of chunk dictionaries.
        """
        if not chunks:
            return

        ids = [chunk["chunk_id"] for chunk in chunks]
        contents = [chunk["content"] for chunk in chunks]
        
        # Embed the documents
        embeddings = embedder.embed_documents(contents)

        # Prepare metadata for ChromaDB (must be strings, numbers, or booleans)
        metadatas = []
        for chunk in chunks:
            metadata = {k: v for k, v in chunk.items() if k != 'content' and isinstance(v, (str, int, float, bool))}
            metadatas.append(metadata)

        self.collection.add(
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Added {len(chunks)} documents to ChromaDB collection '{self.collection.name}'.")

    def _format_results(self, results: Dict[str, Any]) -> List[Tuple[Dict[str, Any], float]]:
        """Helper to format ChromaDB results into the standard format."""
        formatted_results = []
        if not results or not results.get('ids'):
            return formatted_results

        ids = results['ids'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        documents = results['documents'][0]

        for i in range(len(ids)):
            chunk_data = metadatas[i]
            chunk_data['content'] = documents[i]
            score = 1 - distances[i]  # Convert distance to similarity score
            formatted_results.append((chunk_data, score))
            
        return formatted_results

    def similarity_search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Performs a similarity search in ChromaDB.

        Args:
            query (str): The query text.
            top_k (int): The number of top results to return.

        Returns:
            List[Tuple[Dict[str, Any], float]]: A list of (chunk, score) tuples.
        """
        query_embedding = embedder.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return self._format_results(results)

    def metadata_filtered_search(self, query: str, filter_metadata: Dict[str, Any], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Performs a similarity search with metadata filtering in ChromaDB.

        Args:
            query (str): The query text.
            filter_metadata (Dict[str, Any]): The metadata to filter by.
            top_k (int): The number of top results to return.

        Returns:
            List[Tuple[Dict[str, Any], float]]: A list of (chunk, score) tuples.
        """
        query_embedding = embedder.embed_query(query)
        
        # ChromaDB's where filter
        where_filter = filter_metadata

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter
        )
        return self._format_results(results)

    def get_collection_size(self) -> int:
        """
        Returns the number of items in the collection.
        """
        return self.collection.count()

if __name__ == '__main__':
    # Example Usage
    # Ensure you have run the previous examples to have some data
    from economic_graphrag.chunking.chunker import chunk_documents
    from economic_graphrag.ingestion.pdf_loader import load_pdf
    import shutil

    # Clean up previous DB for a fresh start
    if Path(settings.CHROMA_PATH).exists():
        shutil.rmtree(settings.CHROMA_PATH)
        print(f"Removed old ChromaDB at {settings.CHROMA_PATH}")

    # 1. Initialize ChromaVectorStore
    vector_store = ChromaVectorStore()
    print(f"Initial collection size: {vector_store.get_collection_size()}")

    # 2. Load and chunk documents
    dummy_pdf_path = settings.PDF_DIR / "dummy_report.pdf"
    if not dummy_pdf_path.exists():
        print("Dummy PDF not found. Please run pdf_loader.py first.")
    else:
        docs = load_pdf(dummy_pdf_path)
        chunks = chunk_documents(docs)

        # 3. Add documents to the vector store
        if chunks:
            vector_store.add_documents(chunks)
            print(f"New collection size: {vector_store.get_collection_size()}")

            # 4. Perform a similarity search
            query = "What was the GDP growth?"
            search_results = vector_store.similarity_search(query, top_k=1)
            print(f"\n--- Similarity Search Results for '{query}' ---")
            if search_results:
                for chunk, score in search_results:
                    print(f"Score: {score:.4f}")
                    print(f"Content: {chunk['content']}")
                    print("-" * 10)
            else:
                print("No results found.")

            # 5. Perform a metadata filtered search
            # This will likely not find anything unless metadata is set, but shows the principle
            filter_query = "GDP"
            metadata_filter = {"title": "Dummy GDP Report"} # Example filter
            filtered_results = vector_store.metadata_filtered_search(filter_query, metadata_filter, top_k=1)
            print(f"\n--- Metadata Filtered Search Results for '{filter_query}' with filter {metadata_filter} ---")
            if filtered_results:
                for chunk, score in filtered_results:
                    print(f"Score: {score:.4f}")
                    print(f"Content: {chunk['content']}")
                    print(f"Metadata: {chunk}")
                    print("-" * 10)
            else:
                print("No results found with the specified filter.")
        else:
            print("No chunks to add to the vector store.")
