# economic_graphrag/graph/entity_extractor.py
import spacy
from typing import List, Dict, Any, Set, Tuple

from economic_graphrag.llm.ollama_client import OllamaClient

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class EntityExtractor:
    """
    Extracts entities and relationships from text using a combination of
    spaCy NER and an LLM.
    """

    def __init__(self):
        self.ollama_client = OllamaClient()

    def extract_entities_spacy(self, text: str) -> Dict[str, Set[str]]:
        """
        Extracts entities using spaCy's NER.

        Args:
            text (str): The text to process.

        Returns:
            Dict[str, Set[str]]: A dictionary mapping entity types (e.g., 'GPE', 'ORG')
                                 to a set of extracted entity names.
        """
        doc = nlp(text)
        entities = {"Country": set(), "Institution": set()}
        for ent in doc.ents:
            if ent.label_ == "GPE":  # Geopolitical Entity (Countries, Cities, States)
                entities["Country"].add(ent.text)
            elif ent.label_ == "ORG":  # Organization
                entities["Institution"].add(ent.text)
        return entities

    def extract_with_llm(self, chunk_content: str) -> Dict[str, Any]:
        """
        Uses an LLM to extract more complex entities and relationships.

        Args:
            chunk_content (str): The text content of a chunk.

        Returns:
            Dict[str, Any]: A dictionary containing extracted entities and relationships.
                            Example: {'entities': {'Indicator': ['GDP'], 'EconomicConcept': ['growth']},
                                      'relationships': [['GDP', 'HAS_TREND', 'positive']]}
        """
        prompt = f"""
        From the following text, extract economic indicators, concepts, and relationships.
        Respond ONLY with a single valid JSON object with two keys: "entities" and "relationships".
        - "entities" should be a dictionary where keys are one of ["Indicator", "EconomicConcept", "Report", "Source", "Forecast", "Region"] and values are a list of strings.
        - "relationships" should be a list of triplets, where each triplet is [subject, relationship_type, object].
          Relationship types should be one of: ["HAS_INDICATOR", "REPORTED_IN", "MENTIONS", "PUBLISHED_BY", "BELONGS_TO", "CORRELATES_WITH", "AFFECTS", "FORECASTS", "DERIVED_FROM", "HAS_SOURCE"].

        Text: "{chunk_content}"

        Example Response:
        {{
            "entities": {{
                "Indicator": ["GDP", "inflation"],
                "EconomicConcept": ["economic growth"]
            }},
            "relationships": [
                ["economic growth", "CORRELATES_WITH", "GDP"]
            ]
        }}
        """
        
        response = self.ollama_client.generate(prompt, format="json")
        
        if "error" in response or 'response' not in response:
            print(f"LLM extraction failed: {response.get('error', 'No response field')}")
            return {"entities": {}, "relationships": []}
            
        try:
            # The actual response from the model is a string, which needs to be parsed as JSON again
            llm_result = json.loads(response['response'])
            # Basic validation
            if "entities" not in llm_result or "relationships" not in llm_result:
                return {"entities": {}, "relationships": []}
            return llm_result
        except (json.JSONDecodeError, TypeError):
            print(f"Failed to decode LLM response JSON: {response.get('response')}")
            return {"entities": {}, "relationships": []}

    def extract_all(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrates the extraction process using all available methods.

        Args:
            chunk (Dict[str, Any]): The chunk dictionary, containing content and metadata.

        Returns:
            Dict[str, Any]: A dictionary containing all extracted entities and relationships.
        """
        content = chunk.get("content", "")
        
        # 1. spaCy extraction
        spacy_entities = self.extract_entities_spacy(content)
        
        # 2. LLM extraction
        llm_extracts = self.extract_with_llm(content)
        llm_entities = llm_extracts.get("entities", {})
        relationships = llm_extracts.get("relationships", [])

        # 3. Combine entities, ensuring sets for deduplication
        all_entities = {
            "Country": set(), "Institution": set(), "Indicator": set(),
            "EconomicConcept": set(), "Report": set(), "Source": set(),
            "Forecast": set(), "Region": set()
        }

        # Add spaCy entities
        all_entities["Country"].update(spacy_entities.get("Country", set()))
        all_entities["Institution"].update(spacy_entities.get("Institution", set()))

        # Add LLM entities
        for entity_type, entity_list in llm_entities.items():
            if entity_type in all_entities:
                all_entities[entity_type].update(entity_list)
        
        # Add entities from chunk metadata
        if chunk.get("country"): all_entities["Country"].add(chunk["country"])
        if chunk.get("indicator"): all_entities["Indicator"].add(chunk["indicator"])
        if chunk.get("source"): all_entities["Source"].add(chunk["source"])
        if chunk.get("title"): all_entities["Report"].add(chunk["title"])


        # Convert sets back to lists for the final output
        final_entities = {k: list(v) for k, v in all_entities.items() if v}

        return {
            "entities": final_entities,
            "relationships": relationships
        }

if __name__ == '__main__':
    import json
    extractor = EntityExtractor()
    
    test_chunk = {
        "content": "According to a report by the International Monetary Fund (IMF), the GDP of the United States is projected to grow by 2.1% in 2024. This forecast is influenced by current interest rates. The report is titled 'World Economic Outlook'.",
        "country": "United States",
        "indicator": "GDP Growth",
        "source": "IMF WEO Report",
        "title": "World Economic Outlook"
    }

    print("--- Testing spaCy Extraction ---")
    spacy_results = extractor.extract_entities_spacy(test_chunk["content"])
    print(spacy_results)

    print("\n--- Testing LLM Extraction ---")
    # This can be slow and requires Ollama to be running
    try:
        llm_results = extractor.extract_with_llm(test_chunk["content"])
        print(json.dumps(llm_results, indent=2))
    except Exception as e:
        print(f"LLM extraction failed. Is Ollama running? Error: {e}")

    print("\n--- Testing Combined Extraction ---")
    try:
        combined_results = extractor.extract_all(test_chunk)
        print(json.dumps(combined_results, indent=2))
    except Exception as e:
        print(f"Combined extraction failed. Is Ollama running? Error: {e}")
