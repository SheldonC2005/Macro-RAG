# economic_graphrag/agents/graph_agent.py
# This file is conceptually part of the LangGraph workflow.
# The logic for the graph agent will be defined as a node in the LangGraph state machine.
# For clarity and modularity, we can define the agent's core function here.

from typing import Dict, Any, List
from economic_graphrag.retrieval.graph_retriever import GraphRetriever

class GraphAgent:
    """
    An agent responsible for retrieving information from the knowledge graph.
    """

    def __init__(self, graph_retriever: GraphRetriever):
        self.graph_retriever = graph_retriever

    def retrieve(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        The entry point for this agent in the LangGraph workflow.

        Args:
            state (Dict[str, Any]): The current state of the graph, containing the query and plan.

        Returns:
            Dict[str, Any]: A dictionary with the graph search results to be merged back into the state.
        """
        print("--- Executing Graph Agent ---")
        query = state["query"]
        query_type = state.get("plan", {}).get("query_type")
        
        graph_results = self.graph_retriever.retrieve(query, query_type)
        
        print(f"Found {len(graph_results)} results from graph search.")
        
        return {"graph_results": graph_results}

if __name__ == '__main__':
    # This demonstrates the agent's function outside of a LangGraph workflow.
    
    # Mock retriever for standalone testing
    class MockGraphRetriever:
        def retrieve(self, query, query_type):
            if "Inflation" in query:
                return [{'subject': 'Inflation', 'relationship': 'AFFECTS', 'object': 'Interest Rates'}]
            return []

    # 1. Initialize agent with mock retriever
    retriever = MockGraphRetriever()
    agent = GraphAgent(retriever)

    # 2. Simulate a state dictionary
    initial_state = {
        "query": "What is the effect of Inflation?",
        "plan": {
            "query_type": "RelationshipQuery",
            "retrieval_strategy": "graph"
        },
        "vector_results": [],
        "graph_results": []
    }

    # 3. Execute the agent's function
    result_update = agent.retrieve(initial_state)

    # 4. Print the results
    print("\n--- Graph Agent Results ---")
    print(result_update)

    # 5. Show how the state would be updated
    initial_state.update(result_update)
    print("\n--- Updated State ---")
    print(initial_state)
