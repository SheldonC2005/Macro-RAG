# economic_graphrag/agents/planner_agent.py
from typing import Dict, Any

from economic_graphrag.llm.gemini_client import GeminiClient
from economic_graphrag.llm.json_utils import LLMJSONParseError, parse_json_object

class PlannerAgent:
    """
    An agent responsible for classifying a user's query and creating an
    execution plan.
    """

    def __init__(self):
        self.llm_client = GeminiClient()

    def plan(self, query: str) -> Dict[str, Any]:
        """
        Classifies the query and determines the retrieval strategy.

        Args:
            query (str): The user's query.

        Returns:
            Dict[str, Any]: An execution plan.
                            Example: {'query': '...', 'query_type': 'ComparisonQuery',
                                      'retrieval_strategy': 'hybrid'}
        """
        prompt = f"""
        You are a macroeconomic expert agent. Your task is to classify the user's query into one of the following categories:
        - DefinitionQuery: Asks for the definition of a concept. (e.g., "What is GDP?")
        - TrendQuery: Asks about the trend of an indicator over time. (e.g., "What was the inflation trend in 2023?")
        - ComparisonQuery: Compares indicators across different entities. (e.g., "Compare India and China inflation")
        - ForecastQuery: Asks for future projections. (e.g., "What is the GDP forecast for next year?")
        - RelationshipQuery: Asks about the relationship between concepts. (e.g., "How does interest rate affect unemployment?")
        - ResearchQuery: A general, open-ended research question.

        Based on the query, also decide the best retrieval strategy:
        - 'vector': For definition or general research questions.
        - 'graph': For relationship or comparison questions where entities are clear.
        - 'hybrid': For most questions, to get the best of both worlds.

        Respond ONLY with a single valid JSON object containing "query_type" and "retrieval_strategy".

        Query: "{query}"

        Example Response:
        {{
            "query_type": "ComparisonQuery",
            "retrieval_strategy": "hybrid"
        }}
        """
        
        response = self.llm_client.generate(prompt, format="json")

        plan = {
            "query": query,
            "query_type": "ResearchQuery",  # Default
            "retrieval_strategy": "hybrid" # Default
        }

        if "error" in response or 'response' not in response:
            print(f"Planner agent failed: {response.get('error', 'No response field')}")
            return plan # Return default plan

        try:
            llm_result = parse_json_object(response['response'])
            plan['query_type'] = llm_result.get('query_type', plan['query_type'])
            plan['retrieval_strategy'] = llm_result.get('retrieval_strategy', plan['retrieval_strategy'])
        except LLMJSONParseError:
            print(f"Failed to decode planner response JSON: {response.get('response')}")
            # Return default plan on failure
        
        return plan

if __name__ == '__main__':
    import json
    planner = PlannerAgent()

    queries_to_test = [
        "What is the definition of Consumer Price Index?",
        "Compare the GDP of the United States and Germany.",
        "What is the relationship between unemployment and inflation?",
        "What did the latest IMF report say about global growth?"
    ]

    for q in queries_to_test:
        print(f"--- Planning for query: '{q}' ---")
        try:
            execution_plan = planner.plan(q)
            print(json.dumps(execution_plan, indent=2))
        except Exception as e:
            print(f"Planning failed. Is Gemini configured and reachable? Error: {e}")
        print("-" * 20)
