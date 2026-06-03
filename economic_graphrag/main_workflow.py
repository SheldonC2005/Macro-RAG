# economic_graphrag/main_workflow.py
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

from economic_graphrag.agents.planner_agent import PlannerAgent
from economic_graphrag.agents.vector_agent import VectorAgent
from economic_graphrag.agents.graph_agent import GraphAgent
from economic_graphrag.agents.answer_agent import AnswerAgent
from economic_graphrag.retrieval.vector_retriever import VectorRetriever
from economic_graphrag.retrieval.graph_retriever import GraphRetriever
from economic_graphrag.vector_store.chroma_manager import ChromaVectorStore

# Define the state for our LangGraph
class GraphState(TypedDict):
    query: str
    plan: Dict[str, Any]
    vector_results: List[Dict[str, Any]]
    graph_results: List[Dict[str, Any]]
    final_answer: Dict[str, Any]

class MainWorkflow:
    """
    Orchestrates the entire RAG workflow using LangGraph.
    """
    def __init__(self):
        # Initialize components
        self.planner = PlannerAgent()
        
        # Retrievers
        vector_store = ChromaVectorStore()
        vector_retriever = VectorRetriever(vector_store)
        graph_retriever = GraphRetriever()
        
        # Agents that use retrievers
        self.vector_agent = VectorAgent(vector_retriever)
        self.graph_agent = GraphAgent(graph_retriever)
        self.answer_agent = AnswerAgent()
        
        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("planner", self.run_planner)
        workflow.add_node("vector_retriever", self.run_vector_retriever)
        workflow.add_node("graph_retriever", self.run_graph_retriever)
        workflow.add_node("answer_generator", self.run_answer_generator)

        # Define the edges
        workflow.set_entry_point("planner")
        
        # Conditional branching after planning
        workflow.add_conditional_edges(
            "planner",
            self.decide_retrieval_strategy,
            {
                "vector": "vector_retriever",
                "graph": "graph_retriever",
                "hybrid": "vector_retriever" # Start with vector, then go to graph
            }
        )
        
        # After retrieval, generate the answer
        workflow.add_edge("vector_retriever", "graph_retriever") # In hybrid, vector runs then graph
        workflow.add_edge("graph_retriever", "answer_generator")
        workflow.add_edge("answer_generator", END)

        return workflow.compile()

    # --- Node execution functions ---

    def run_planner(self, state: GraphState) -> GraphState:
        print("--- Running Planner ---")
        query = state['query']
        plan = self.planner.plan(query)
        return {"plan": plan}

    def run_vector_retriever(self, state: GraphState) -> GraphState:
        print("--- Running Vector Retriever ---")
        return self.vector_agent.retrieve(state)

    def run_graph_retriever(self, state: GraphState) -> GraphState:
        print("--- Running Graph Retriever ---")
        return self.graph_agent.retrieve(state)

    def run_answer_generator(self, state: GraphState) -> GraphState:
        print("--- Generating Final Answer ---")
        query = state['query']
        context = {
            "vector_results": state.get("vector_results", []),
            "graph_results": state.get("graph_results", [])
        }
        final_answer = self.answer_agent.generate_answer(query, context)
        return {"final_answer": final_answer}

    # --- Conditional edge logic ---

    def decide_retrieval_strategy(self, state: GraphState) -> str:
        print(f"--- Deciding strategy: {state['plan']['retrieval_strategy']} ---")
        return state['plan']['retrieval_strategy']

    def execute(self, query: str) -> Dict[str, Any]:
        """
        Executes the RAG workflow for a given query.

        Args:
            query (str): The user's query.

        Returns:
            Dict[str, Any]: The final state of the graph, including the answer.
        """
        initial_state = {"query": query, "plan": {}, "vector_results": [], "graph_results": [], "final_answer": {}}
        final_state = self.graph.invoke(initial_state)
        return final_state

if __name__ == '__main__':
    # This is a full end-to-end example. It requires all components to be set up,
    # including populated vector and graph databases, and a running Ollama instance.
    
    print("--- Initializing Main Workflow ---")
    workflow = MainWorkflow()
    
    # Example query
    query = "What is the relationship between GDP and inflation in the United States?"
    
    print(f"\n--- Executing workflow for query: '{query}' ---")
    
    try:
        # The execute method will stream print statements from the nodes
        result = workflow.execute(query)
        
        print("\n--- Workflow Complete ---")
        print("\n--- Final Answer ---")
        import json
        print(json.dumps(result.get('final_answer'), indent=2))
        
    except Exception as e:
        print(f"\nAn error occurred during workflow execution: {e}")
        print("Please ensure all services (Ollama, Neo4j) are running and data has been ingested.")
