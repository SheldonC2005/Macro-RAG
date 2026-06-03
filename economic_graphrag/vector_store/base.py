# economic_graphrag/vector_store/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

class BaseVectorStore(ABC):
    """
    Abstract Base Class for a vector store.
    This defines the interface for future vector store implementations (e.g., Chroma, Milvus).
    """

    @abstractmethod
    def add_documents(self, chunks: List[Dict[str, Any]]):
        """
        Adds documents (chunks) to the vector store.

        Args:
            chunks (List[Dict[str, Any]]): A list of chunk dictionaries. Each chunk
                                           must have 'content' and 'chunk_id'.
        """
        pass

    @abstractmethod
    def similarity_search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Performs a similarity search for a given query.

        Args:
            query (str): The query text.
            top_k (int): The number of top results to return.

        Returns:
            List[Tuple[Dict[str, Any], float]]: A list of tuples, where each tuple contains
                                                a result document (chunk) and its similarity score.
        """
        pass

    @abstractmethod
    def metadata_filtered_search(self, query: str, filter_metadata: Dict[str, Any], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Performs a similarity search with metadata filtering.

        Args:
            query (str): The query text.
            filter_metadata (Dict[str, Any]): A dictionary of metadata to filter by.
            top_k (int): The number of top results to return.

        Returns:
            List[Tuple[Dict[str, Any], float]]: A list of tuples, where each tuple contains
                                                a result document (chunk) and its similarity score.
        """
        pass

    @abstractmethod
    def get_collection_size(self) -> int:
        """
        Returns the number of items in the collection.

        Returns:
            int: The number of items.
        """
        pass
