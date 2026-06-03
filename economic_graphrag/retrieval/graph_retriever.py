# economic_graphrag/retrieval/graph_retriever.py
from typing import List, Dict, Any

from economic_graphrag.graph.neo4j_manager import neo4j_manager

class GraphRetriever:
    """
    A class to handle retrieval of information from the Neo4j knowledge graph.
    """

    def retrieve(self, query: str, query_type: str) -> List[Dict[str, Any]]:
        """
        Retrieves graph data based on a natural language query and its type.
        This is a simplified implementation; a production system would use LLM to
        generate Cypher queries. Here, we use predefined query patterns.

        Args:
            query (str): The natural language query.
            query_type (str): The classified type of the query (e.g., 'Relationship', 'Trend').

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the retrieved paths or nodes.
        """
        # For this prototype, we'll use simple keyword matching and query patterns.
        # A more advanced version would use an LLM to generate Cypher.
        
        # Simple entity extraction from query
        entities = self._extract_entities_from_query(query)
        
        cypher_query = ""
        params = {}

        if query_type == "RelationshipQuery" and len(entities) >= 1:
            entity_name = entities[0]
            cypher_query = """
            MATCH (a {name: $entity_name})-[r]-(b)
            RETURN a.name AS subject, type(r) AS relationship, b.name AS object
            LIMIT 10
            """
            params = {"entity_name": entity_name}
        
        elif query_type == "ComparisonQuery" and len(entities) >= 2:
            entity1, entity2 = entities[0], entities[1]
            cypher_query = """
            MATCH (a {name: $entity1})-[r]-(p)
            MATCH (b {name: $entity2})-[r]-(p)
            RETURN a.name AS entity1, b.name AS entity2, p.name AS common_point, type(r) as relationship
            LIMIT 10
            """
            params = {"entity1": entity1, "entity2": entity2}

        elif query_type == "DefinitionQuery" and len(entities) == 1:
            entity_name = entities[0]
            cypher_query = """
            MATCH (n {name: $entity_name})
            RETURN n.name as name, labels(n) as labels, properties(n) as properties
            LIMIT 1
            """
            params = {"entity_name": entity_name}

        else: # Default fallback
            if entities:
                entity_name = entities[0]
                cypher_query = """
                MATCH (a {name: $entity_name})-[r]-(b)
                RETURN a.name AS subject, type(r) AS relationship, b.name AS object
                LIMIT 5
                """
                params = {"entity_name": entity_name}

        if cypher_query:
            try:
                results = neo4j_manager.execute_query(cypher_query, params)
                return results
            except Exception as e:
                print(f"Error executing graph query: {e}")
                return []
        
        return []

    def _extract_entities_from_query(self, query: str) -> List[str]:
        """
        A very basic method to extract potential entities from a query.
        It capitalizes words, assuming entities are proper nouns.
        A real implementation would use a more sophisticated NER approach.
        """
        # This is a placeholder for more advanced NER on the query itself.
        return [word for word in query.replace("?", "").split() if word[0].isupper()]


if __name__ == '__main__':
    # Example Usage
    # This requires a populated Neo4j database.
    
    # 1. Setup: Add some data to Neo4j for testing
    try:
        neo4j_manager.create_constraints()
        setup_queries = [
            "MERGE (c:Country {name: 'India'})",
            "MERGE (c:Country {name: 'China'})",
            "MERGE (i:Indicator {name: 'Inflation'})",
            "MERGE (i2:Indicator {name: 'GDP'})",
            "MERGE (c:Country {name: 'India'})-[:HAS_INDICATOR]->(i:Indicator {name: 'Inflation'})",
            "MERGE (c:Country {name: 'China'})-[:HAS_INDICATOR]->(i:Indicator {name: 'Inflation'})",
            "MERGE (c:Country {name: 'India'})-[:HAS_INDICATOR]->(i2:Indicator {name: 'GDP'})",
            "MERGE (r:EconomicConcept {name: 'Interest Rates'})",
            "MERGE (i:Indicator {name: 'Inflation'})-[:AFFECTS]->(r:EconomicConcept {name: 'Interest Rates'})",
        ]
        for q in setup_queries:
            neo4j_manager.execute_write(q)
        print("Test data added to Neo4j.")

        # 2. Initialize the retriever
        graph_retriever = GraphRetriever()

        # 3. Test a relationship query
        query1 = "What affects Inflation?"
        print(f"\n--- Retrieving for query: '{query1}' (RelationshipQuery) ---")
        results1 = graph_retriever.retrieve(query1, "RelationshipQuery")
        if results1:
            for res in results1:
                print(res)
        else:
            print("No results found.")

        # 4. Test a comparison query
        query2 = "Compare India and China"
        print(f"\n--- Retrieving for query: '{query2}' (ComparisonQuery) ---")
        results2 = graph_retriever.retrieve(query2, "ComparisonQuery")
        if results2:
            for res in results2:
                print(res)
        else:
            print("No results found.")
            
        # 5. Test a definition query
        query3 = "What is GDP?"
        print(f"\n--- Retrieving for query: '{query3}' (DefinitionQuery) ---")
        results3 = graph_retriever.retrieve(query3, "DefinitionQuery")
        if results3:
            for res in results3:
                print(res)
        else:
            print("No results found.")

    except ConnectionError as e:
        print(f"Could not connect to Neo4j. Please ensure it's running. Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up test data
        print("\n--- Cleaning up test data ---")
        cleanup_queries = [
            "MATCH (n) WHERE n.name IN ['India', 'China', 'Inflation', 'GDP', 'Interest Rates'] DETACH DELETE n"
        ]
        for q in cleanup_queries:
            neo4j_manager.execute_write(q)
        print("Test data cleaned up.")
        neo4j_manager.close()
