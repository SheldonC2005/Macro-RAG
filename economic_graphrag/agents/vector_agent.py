# economic_graphrag/agents/vector_agent.py
# This file is conceptually part of the LangGraph workflow.
# The logic for the vector agent will be defined as a node in the LangGraph state machine.

from typing import Dict, Any, List
from economic_graphrag.retrieval.vector_retriever import VectorRetriever

class VectorAgent:
    """
    An agent responsible for retrieving information from the vector store.
    """

    def __init__(self, vector_retriever: VectorRetriever):
        self.vector_retriever = vector_retriever

    def retrieve(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        The entry point for this agent in the LangGraph workflow.

        Args:
            state (Dict[str, Any]): The current state of the graph, containing the query.

        Returns:
            Dict[str, Any]: A dictionary with the vector search results to be merged back into the state.
        """
        print("--- Executing Vector Agent ---")
        query = state["query"]
        
        # In a more complex scenario, the plan might specify top_k or filters
        top_k = 5
        
        vector_results = self.vector_retriever.retrieve(query, top_k)
        
        print(f"Found {len(vector_results)} results from vector search.")
        
        return {"vector_results": vector_results}

if __name__ == '__main__':
    # This demonstrates the agent's function outside of a LangGraph workflow.
    
    # Mock retriever for standalone testing
    class MockVectorRetriever:
        def retrieve(self, query, top_k, filter_metadata=None):
            return [
                {'content': f"Vector result for '{query}'", 'score': 0.9},
                {'content': f"Another vector result for '{query}'", 'score': 0.8}
            ]

    # 1. Initialize agent with mock retriever
    retriever = MockVectorRetriever()
    agent = VectorAgent(retriever)

    # 2. Simulate a state dictionary
    initial_state = {
        "query": "What is GDP?",
        "plan": {
            "query_type": "DefinitionQuery",
            "retrieval_strategy": "vector"
        },
        "vector_results": [],
        "graph_results": []
    }

    # 3. Execute the agent's function
    result_update = agent.retrieve(initial_state)

    # 4. Print the results
    print("\n--- Vector Agent Results ---")
    print(result_update)

    # 5. Show how the state would be updated
    initial_state.update(result_update)
    print("\n--- Updated State ---")
    print(initial_state)
