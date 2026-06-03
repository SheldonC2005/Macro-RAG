# economic_graphrag/agents/answer_agent.py
from typing import Dict, Any, List

from economic_graphrag.llm.gemini_client import GeminiClient
from economic_graphrag.llm.json_utils import LLMJSONParseError, parse_json_object

class AnswerAgent:
    """
    An agent that synthesizes context from retrievers to generate a final answer.
    """

    def __init__(self):
        self.llm_client = GeminiClient()

    def generate_answer(self, query: str, context: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Generates a final answer based on the query and the retrieved context.

        Args:
            query (str): The user's original query.
            context (Dict[str, List[Dict[str, Any]]]): The retrieved context from vector and graph searches.

        Returns:
            Dict[str, Any]: A dictionary containing the answer, sources, and explainability details.
        """
        
        # Prepare the context for the LLM prompt
        vector_context_str = self._format_vector_context(context.get("vector_results", []))
        graph_context_str = self._format_graph_context(context.get("graph_results", []))

        prompt = f"""
        You are an expert macroeconomic analyst. Your task is to answer the user's query based on the provided context.
        Your answer must be comprehensive, accurate, and directly supported by the context.
        Cite your sources clearly.

        **User Query:**
        {query}

        **Context from Vector Search (Unstructured Documents):**
        {vector_context_str}

        **Context from Graph Search (Structured Relationships):**
        {graph_context_str}

        **Instructions:**
        1. Formulate a clear and concise answer to the user's query.
        2. Base your answer *only* on the provided context. Do not use any external knowledge.
        3. If the context is insufficient, state that you cannot answer the question with the given information.
        4. List the sources you used from the context. A source is the 'source' or 'title' from the vector context.
        5. Explain the reasoning by mentioning the key graph relationships or document chunks that support your answer.

        Respond ONLY with a single valid JSON object with the following keys:
        - "answer": Your detailed answer to the query.
        - "sources": A list of source document titles/names you used.
        - "explainability": A brief explanation of how you arrived at the answer, referencing the context.

        Example Response:
        {{
            "answer": "The GDP of Testland showed strong growth in 2023. This growth is correlated with low inflation during the same period.",
            "sources": ["Dummy GDP Report", "Economic Trends 2023"],
            "explainability": "The answer was synthesized from a document chunk mentioning GDP growth and a graph relationship showing a correlation between GDP and low inflation."
        }}
        """

        response = self.llm_client.generate(prompt, format="json")

        if "error" in response or 'response' not in response:
            return {"answer": "Failed to generate an answer.", "sources": [], "explainability": response.get("error", "No response from LLM.")}

        try:
            llm_result = parse_json_object(response['response'])
            return llm_result
        except LLMJSONParseError:
            return {"answer": "Failed to parse the generated answer.", "sources": [], "explainability": f"Raw response: {response.get('response')}"}

    def _format_vector_context(self, vector_results: List[Dict[str, Any]]) -> str:
        if not vector_results:
            return "No relevant documents found."
        
        formatted_str = ""
        for i, res in enumerate(vector_results):
            formatted_str += f"Chunk {i+1} (Source: {res.get('source', 'N/A')}, Score: {res.get('score', 0):.4f}):\n"
            formatted_str += f"\"{res.get('content', '')}\"\n\n"
        return formatted_str

    def _format_graph_context(self, graph_results: List[Dict[str, Any]]) -> str:
        if not graph_results:
            return "No relevant graph relationships found."
            
        formatted_str = "The following relationships were found:\n"
        for res in graph_results:
            subj = res.get('subject', 'N/A')
            rel = res.get('relationship', 'N/A')
            obj = res.get('object', 'N/A')
            formatted_str += f"- ({subj}) -[{rel}]-> ({obj})\n"
        return formatted_str

if __name__ == '__main__':
    import json
    answer_agent = AnswerAgent()

    # Simulate context
    mock_context = {
        "vector_results": [
            {
                "content": "In its 2024 report, the IMF noted that high inflation in developing countries often leads to central banks raising interest rates.",
                "source": "IMF Report 2024",
                "score": 0.91
            }
        ],
        "graph_results": [
            {
                "subject": "Inflation",
                "relationship": "AFFECTS",
                "object": "Interest Rates"
            }
        ]
    }

    query = "How does inflation impact interest rates according to the IMF?"

    print(f"--- Generating answer for query: '{query}' ---")
    try:
        final_answer = answer_agent.generate_answer(query, mock_context)
        print(json.dumps(final_answer, indent=2))
    except Exception as e:
        print(f"Answer generation failed. Is Gemini configured and reachable? Error: {e}")
