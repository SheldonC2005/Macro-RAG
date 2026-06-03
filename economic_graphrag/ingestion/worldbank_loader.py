# economic_graphrag/ingestion/worldbank_loader.py
import requests
import pandas as pd
from typing import List, Dict, Any
import uuid

# G20 countries for initial data loading
G20_COUNTRIES = {
    "ARG": "Argentina", "AUS": "Australia", "BRA": "Brazil", "CAN": "Canada",
    "CHN": "China", "FRA": "France", "DEU": "Germany", "IND": "India",
    "IDN": "Indonesia", "ITA": "Italy", "JPN": "Japan", "MEX": "Mexico",
    "RUS": "Russia", "SAU": "Saudi Arabia", "ZAF": "South Africa",
    "KOR": "South Korea", "TUR": "Turkey", "GBR": "United Kingdom", "USA": "United States"
}

# Core indicators to fetch
INDICATORS = {
    "NY.GDP.MKTP.CD": "GDP (current US$)",
    "FP.CPI.TOTL.ZG": "Inflation, consumer prices (annual %)",
    "SP.POP.TOTL": "Population, total",
    "NE.TRD.GNFS.ZS": "Trade (% of GDP)"
}

def get_worldbank_data(country_code: str, indicator_code: str, start_year: int = 2000, end_year: int = 2023) -> List[Dict[str, Any]]:
    """
    Fetches data for a specific country and indicator from the World Bank API.
    """
    url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}"
    params = {
        "format": "json",
        "date": f"{start_year}:{end_year}",
        "per_page": 1000  # Fetch all data in one go
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if data and len(data) > 1 and data[1]:
            records = []
            for entry in data[1]:
                if entry['value'] is not None:
                    records.append({
                        "country": entry['country']['value'],
                        "indicator": entry['indicator']['value'],
                        "year": int(entry['date']),
                        "value": float(entry['value'])
                    })
            return records
    except requests.exceptions.RequestException as e:
        print(f"Error fetching World Bank data for {country_code}/{indicator_code}: {e}")
    return []

def load_worldbank_data() -> List[Dict[str, Any]]:
    """
    Loads a core set of World Bank data for G20 countries and formats it
    into the document structure for the RAG system.
    """
    all_documents = []
    for country_code, country_name in G20_COUNTRIES.items():
        for indicator_code, indicator_name in INDICATORS.items():
            print(f"Fetching {indicator_name} for {country_name}...")
            records = get_worldbank_data(country_code, indicator_code)
            
            if records:
                # Create a single document per country-indicator pair
                df = pd.DataFrame(records)
                content = f"Historical data for {indicator_name} in {country_name}:\n\n"
                content += df.to_string(index=False)
                
                doc = {
                    "document_id": str(uuid.uuid4()),
                    "title": f"{indicator_name} - {country_name}",
                    "source": "World Bank API",
                    "content": content,
                    "publication_date": None, # This is time-series data
                    "country": country_name,
                    "indicator": indicator_name,
                }
                all_documents.append(doc)
    
    return all_documents

if __name__ == '__main__':
    print("--- Loading World Bank Data ---")
    # This will make live API calls
    worldbank_documents = load_worldbank_data()
    
    if worldbank_documents:
        print(f"\nSuccessfully loaded {len(worldbank_documents)} documents from the World Bank API.")
        print("\n--- Example Document ---")
        example_doc = worldbank_documents[0]
        print(f"Title: {example_doc['title']}")
        print(f"Source: {example_doc['source']}")
        print(f"Country: {example_doc['country']}")
        print(f"Indicator: {example_doc['indicator']}")
        print("\n--- Content Snippet ---")
        print(example_doc['content'][:300] + "...")
    else:
        print("Failed to load any documents from the World Bank API.")
