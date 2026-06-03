# economic_graphrag/graph/graph_builder.py
from typing import List, Dict, Any

from .neo4j_manager import neo4j_manager
from .entity_extractor import EntityExtractor

class GraphBuilder:
    """
    Builds the knowledge graph by extracting entities and relationships
    from documents and inserting them into Neo4j.
    """

    def __init__(self):
        self.entity_extractor = EntityExtractor()

    def build_graph_from_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Processes a list of chunks to build the graph.

        Args:
            chunks (List[Dict[str, Any]]): A list of chunk dictionaries.
        """
        total_chunks = len(chunks)
        print(f"Starting graph build process for {total_chunks} chunks.")
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{total_chunks}...")
            try:
                extracted_data = self.entity_extractor.extract_all(chunk)
                self._insert_data_into_graph(extracted_data, chunk)
            except Exception as e:
                print(f"Error processing chunk {chunk.get('chunk_id', 'N/A')}: {e}")

        print("Graph build process completed.")

    def _insert_data_into_graph(self, data: Dict[str, Any], chunk: Dict[str, Any]):
        """
        Inserts extracted entities and relationships into Neo4j.

        Args:
            data (Dict[str, Any]): The extracted entities and relationships.
            chunk (Dict[str, Any]): The original chunk for context.
        """
        entities = data.get("entities", {})
        relationships = data.get("relationships", [])

        # Insert entities first
        for entity_type, entity_list in entities.items():
            for entity_name in entity_list:
                # Use MERGE to avoid creating duplicate nodes
                query = f"MERGE (n:{entity_type} {{name: $name}})"
                neo4j_manager.execute_write(query, {"name": entity_name})

        # Insert relationships
        for rel in relationships:
            if len(rel) == 3:
                subj, rel_type, obj = rel
                # Find the types of subject and object from the entities dict
                subj_type = self._find_entity_type(subj, entities)
                obj_type = self._find_entity_type(obj, entities)

                if subj_type and obj_type:
                    query = f"""
                    MATCH (a:{subj_type} {{name: $subj_name}})
                    MATCH (b:{obj_type} {{name: $obj_name}})
                    MERGE (a)-[:{rel_type}]->(b)
                    """
                    parameters = {"subj_name": subj, "obj_name": obj}
                    neo4j_manager.execute_write(query, parameters)

    def _find_entity_type(self, entity_name: str, entities: Dict[str, List[str]]) -> str:
        """
        Finds the label/type of a given entity name from the extracted entities.
        """
        for entity_type, entity_list in entities.items():
            if entity_name in entity_list:
                return entity_type
        return None

if __name__ == '__main__':
    # Example Usage
    # This is a more complex example that requires other components.
    # We will simulate the inputs.
    
    # 1. Initialize GraphBuilder
    graph_builder = GraphBuilder()

    # 2. Simulate some chunks
    simulated_chunks = [
        {
            "chunk_id": "chunk_1",
            "content": "The GDP of Freedonia showed strong growth in 2023, as reported by the World Bank.",
            "country": "Freedonia",
            "source": "World Bank Report 2023"
        },
        {
            "chunk_id": "chunk_2",
            "content": "High inflation in Freedonia affects consumer spending.",
            "country": "Freedonia",
            "source": "Local News"
        }
    ]

    # 3. Simulate entity extraction (to avoid slow LLM calls in this example)
    def mock_extract_all(chunk):
        if chunk["chunk_id"] == "chunk_1":
            return {
                "entities": {
                    "Indicator": ["GDP"],
                    "Country": ["Freedonia"],
                    "Institution": ["World Bank"]
                },
                "relationships": [
                    ["Freedonia", "HAS_INDICATOR", "GDP"]
                ]
            }
        if chunk["chunk_id"] == "chunk_2":
            return {
                "entities": {
                    "Indicator": ["inflation", "consumer spending"],
                    "Country": ["Freedonia"]
                },
                "relationships": [
                    ["inflation", "AFFECTS", "consumer spending"]
                ]
            }
        return {"entities": {}, "relationships": []}

    graph_builder.entity_extractor.extract_all = mock_extract_all
    print("Mocked entity extractor.")

    # 4. Build the graph
    print("\nBuilding graph with simulated data...")
    try:
        # Ensure constraints are set
        neo4j_manager.create_constraints()
        graph_builder.build_graph_from_chunks(simulated_chunks)

        # 5. Verify insertion
        print("\nVerifying data in Neo4j...")
        query = "MATCH (n) RETURN labels(n) AS Labels, n.name AS Name LIMIT 10"
        results = neo4j_manager.execute_query(query)
        print("Nodes found in graph:")
        for res in results:
            print(res)

        query_rels = "MATCH (a)-[r]->(b) RETURN a.name, type(r), b.name LIMIT 10"
        results_rels = neo4j_manager.execute_query(query_rels)
        print("\nRelationships found in graph:")
        for res in results_rels:
            print(res)

        # 6. Clean up
        print("\nCleaning up test data...")
        cleanup_queries = [
            "MATCH (n:Country {name: 'Freedonia'}) DETACH DELETE n",
            "MATCH (n:Indicator {name: 'GDP'}) DETACH DELETE n",
            "MATCH (n:Indicator {name: 'inflation'}) DETACH DELETE n",
            "MATCH (n:Indicator {name: 'consumer spending'}) DETACH DELETE n",
            "MATCH (n:Institution {name: 'World Bank'}) DETACH DELETE n",
        ]
        for q in cleanup_queries:
            neo4j_manager.execute_write(q)
        print("Cleanup complete.")

    except ConnectionError as e:
        print(f"\nCould not connect to Neo4j. Please ensure it's running. Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        neo4j_manager.close()
