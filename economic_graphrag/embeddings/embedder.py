# economic_graphrag/embeddings/embedder.py
from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np

from economic_graphrag.config import settings

class Embedder:
    """
    A singleton class to handle sentence embeddings using sentence-transformers.
    This ensures the model is loaded only once.
    """
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Embedder, cls).__new__(cls)
            try:
                cls._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
                print(f"Embedding model '{settings.EMBEDDING_MODEL_NAME}' loaded successfully.")
            except Exception as e:
                print(f"Error loading embedding model: {e}")
                cls._model = None
        return cls._instance

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of documents.

        Args:
            texts (List[str]): A list of document texts to embed.

        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.
        """
        if self._model is None:
            raise ConnectionError("Embedding model is not loaded.")
        
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single query.

        Args:
            query (str): The query text to embed.

        Returns:
            List[float]: The embedding for the query.
        """
        if self._model is None:
            raise ConnectionError("Embedding model is not loaded.")
            
        embedding = self._model.encode(query, convert_to_numpy=True)
        return embedding.tolist()

# Singleton instance
embedder = Embedder()

if __name__ == '__main__':
    # Example Usage
    try:
        documents_to_embed = [
            "The global economy is projected to grow by 3% in 2024.",
            "Inflation remains a key concern for central banks worldwide.",
            "Geopolitical tensions are creating uncertainty in financial markets."
        ]
        
        query_to_embed = "What is the economic outlook?"

        # Embed documents
        doc_embeddings = embedder.embed_documents(documents_to_embed)
        print(f"Successfully embedded {len(doc_embeddings)} documents.")
        print(f"Dimension of embeddings: {len(doc_embeddings[0])}")

        # Embed a query
        query_embedding = embedder.embed_query(query_to_embed)
        print("\nSuccessfully embedded the query.")
        print(f"Dimension of query embedding: {len(query_embedding)}")

        # Verify singleton pattern
        embedder2 = Embedder()
        print(f"\nIs embedder instance the same? {embedder is embedder2}")

    except ConnectionError as e:
        print(f"Could not run example: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
