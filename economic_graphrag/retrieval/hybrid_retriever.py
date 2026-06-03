# economic_graphrag/retrieval/hybrid_retriever.py
from typing import List, Dict, Any

from .vector_retriever import VectorRetriever
from .graph_retriever import GraphRetriever
from economic_graphrag.config import settings

class HybridRetriever:
    """
    A class to perform hybrid retrieval, combining results from both
    vector and graph retrievers.
    """

    def __init__(self, vector_retriever: VectorRetriever, graph_retriever: GraphRetriever):
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever

    def retrieve(self, query: str, query_type: str, top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Performs hybrid retrieval by querying both vector and graph stores
        and returning a combined, weighted result.

        Args:
            query (str): The user's query.
            query_type (str): The classified type of the query.
            top_k (int): The number of results to fetch from the vector store.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary containing the results
                                             from 'vector_search' and 'graph_search'.
        """
        # Perform retrievals in parallel (conceptually)
        vector_results = self.vector_retriever.retrieve(query, top_k)
        graph_results = self.graph_retriever.retrieve(query, query_type)

        # For this implementation, we will keep the results separate for the
        # Answer Agent to fuse. A more complex implementation could perform
        # re-ranking here based on the fusion weights.
        
        return {
            "vector_results": vector_results,
            "graph_results": graph_results
        }

    def fuse_and_rank(self, hybrid_results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Fuses results from vector and graph search and re-ranks them.
        This is a simple example of fusion logic.

        Args:
            hybrid_results (Dict[str, List[Dict[str, Any]]]): The results from the retrieve method.

        Returns:
            List[Dict[str, Any]]: A single ranked list of context items.
        """
        fused_results = []

        # Add vector results with their scores, weighted
        for res in hybrid_results.get("vector_results", []):
            res['final_score'] = res.get('score', 0) * settings.VECTOR_WEIGHT
            res['retrieval_source'] = 'vector'
            fused_results.append(res)

        # Add graph results. We need to assign a score.
        # For simplicity, we'll give graph results a fixed high score, weighted.
        graph_base_score = 0.9
        for res in hybrid_results.get("graph_results", []):
            res['final_score'] = graph_base_score * settings.GRAPH_WEIGHT
            res['retrieval_source'] = 'graph'
            # To make it compatible with chunk format, we'll format it as a "content" string
            res['content'] = f"Graph Path: {res}"
            fused_results.append(res)

        # Sort by the new final_score
        fused_results.sort(key=lambda x: x['final_score'], reverse=True)

        return fused_results


if __name__ == '__main__':
    # This is an example of how to use the HybridRetriever.
    # It requires mocked or real retrievers.

    class MockVectorRetriever:
        def retrieve(self, query, top_k, filter_metadata=None):
            return [
                {'content': 'Vector result 1 about GDP', 'score': 0.9, 'country': 'Testland'},
                {'content': 'Vector result 2 about inflation', 'score': 0.85, 'country': 'Testland'}
            ]

    class MockGraphRetriever:
        def retrieve(self, query, query_type):
            return [
                {'subject': 'Inflation', 'relationship': 'AFFECTS', 'object': 'Interest Rates'}
            ]

    # 1. Initialize retrievers
    vector_retriever = MockVectorRetriever()
    graph_retriever = MockGraphRetriever()
    hybrid_retriever = HybridRetriever(vector_retriever, graph_retriever)

    # 2. Perform hybrid retrieval
    query = "What is the effect of inflation in Testland?"
    query_type = "RelationshipQuery"
    
    print(f"--- Performing hybrid retrieval for query: '{query}' ---")
    results = hybrid_retriever.retrieve(query, query_type)

    print("\n--- Raw Vector Results ---")
    print(results["vector_results"])
    
    print("\n--- Raw Graph Results ---")
    print(results["graph_results"])

    # 3. Fuse and rank the results
    print("\n--- Fused and Ranked Results ---")
    fused = hybrid_retriever.fuse_and_rank(results)
    for item in fused:
        print(f"Source: {item['retrieval_source']}, Score: {item['final_score']:.4f}")
        print(f"Content: {item['content']}")
        print("-" * 10)
