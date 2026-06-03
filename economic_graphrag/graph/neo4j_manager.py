# economic_graphrag/graph/neo4j_manager.py
from neo4j import GraphDatabase, basic_auth
from typing import List, Dict, Any

from economic_graphrag.config import settings

class Neo4jManager:
    """
    A singleton class to manage the connection to a Neo4j database.
    """
    _instance = None
    _driver = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Neo4jManager, cls).__new__(cls)
            try:
                cls._driver = GraphDatabase.driver(
                    settings.NEO4J_URI,
                    auth=basic_auth(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
                )
                cls._driver.verify_connectivity()
                print("Neo4j connection successful.")
            except Exception as e:
                print(f"Failed to connect to Neo4j: {e}")
                cls._driver = None
        return cls._instance

    def close(self):
        """Closes the Neo4j driver connection."""
        if self._driver is not None:
            self._driver.close()
            print("Neo4j connection closed.")

    def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Executes a Cypher query and returns the results.

        Args:
            query (str): The Cypher query to execute.
            parameters (Dict[str, Any], optional): Parameters for the query. Defaults to None.

        Returns:
            List[Dict[str, Any]]: A list of result records.
        """
        if self._driver is None:
            raise ConnectionError("Neo4j driver not initialized.")
        
        with self._driver.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]

    def execute_write(self, query: str, parameters: Dict[str, Any] = None):
        """
        Executes a write transaction.

        Args:
            query (str): The Cypher query for the write operation.
            parameters (Dict[str, Any], optional): Parameters for the query. Defaults to None.
        """
        if self._driver is None:
            raise ConnectionError("Neo4j driver not initialized.")
            
        with self._driver.session() as session:
            session.write_transaction(self._run_transaction, query, parameters)

    @staticmethod
    def _run_transaction(tx, query, parameters):
        tx.run(query, parameters)

    def create_constraints(self):
        """
        Creates unique constraints on nodes to prevent duplicates.
        This is idempotent and safe to run multiple times.
        """
        constraints = {
            "Country": "name",
            "Indicator": "name",
            "Report": "id",
            "Source": "name",
            "EconomicConcept": "name",
            "Institution": "name",
            "Region": "name",
            "Forecast": "id"
        }
        for label, prop in constraints.items():
            query = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
            try:
                self.execute_write(query)
                print(f"Ensured constraint exists for {label}({prop})")
            except Exception as e:
                print(f"Error creating constraint for {label}({prop}): {e}")


# Singleton instance
neo4j_manager = Neo4jManager()

if __name__ == '__main__':
    try:
        # 1. Create constraints
        print("--- Creating constraints ---")
        neo4j_manager.create_constraints()

        # 2. Example Write Operation (MERGE is idempotent)
        print("\n--- Testing write operation ---")
        write_query = """
        MERGE (c:Country {name: 'Testland'})
        MERGE (i:Indicator {name: 'TestGDP'})
        MERGE (c)-[:HAS_INDICATOR]->(i)
        """
        neo4j_manager.execute_write(write_query)
        print("Successfully executed write query.")

        # 3. Example Read Operation
        print("\n--- Testing read operation ---")
        read_query = "MATCH (c:Country {name: 'Testland'})-[:HAS_INDICATOR]->(i) RETURN c.name AS country, i.name AS indicator"
        results = neo4j_manager.execute_query(read_query)
        
        if results:
            print("Query results:")
            for record in results:
                print(record)
        else:
            print("No results found from test query.")

        # 4. Clean up test data
        print("\n--- Cleaning up test data ---")
        cleanup_query = "MATCH (c:Country {name: 'Testland'}) DETACH DELETE c"
        neo4j_manager.execute_write(cleanup_query)
        cleanup_query_indicator = "MATCH (i:Indicator {name: 'TestGDP'}) DETACH DELETE i"
        neo4j_manager.execute_write(cleanup_query_indicator)
        print("Test data cleaned up.")

    except ConnectionError as e:
        print(f"\nCould not connect to Neo4j. Please ensure the database is running and credentials in config/settings.py are correct.")
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Close the connection
        neo4j_manager.close()
