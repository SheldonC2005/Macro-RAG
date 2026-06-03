# economic_graphrag/ingestion/fred_loader.py
import requests
import pandas as pd
from typing import List, Dict, Any
import uuid

from economic_graphrag.config import settings

# Core FRED series IDs
FRED_SERIES = {
    "CPIAUCSL": "Consumer Price Index for All Urban Consumers",
    "FEDFUNDS": "Effective Federal Funds Rate",
    "UNRATE": "Civilian Unemployment Rate",
    "DGS10": "10-Year Treasury Constant Maturity Rate"
}

def get_fred_data(series_id: str, start_date: str = "2000-01-01") -> pd.DataFrame:
    """
    Fetches data for a specific series from the FRED API.
    """
    api_key = settings.FRED_API_KEY
    if not api_key:
        raise ValueError("FRED_API_KEY not found in environment variables.")

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "observations" in data:
            df = pd.DataFrame(data["observations"])
            df = df[['date', 'value']]
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df['date'] = pd.to_datetime(df['date'])
            return df.dropna()
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching FRED data for {series_id}: {e}")
    return pd.DataFrame()

def load_fred_data() -> List[Dict[str, Any]]:
    """
    Loads a core set of FRED data and formats it into the document
    structure for the RAG system.
    """
    if not settings.FRED_API_KEY:
        print("Skipping FRED data loading: FRED_API_KEY is not set.")
        return []

    all_documents = []
    for series_id, series_name in FRED_SERIES.items():
        print(f"Fetching {series_name} (ID: {series_id})...")
        df = get_fred_data(series_id)
        
        if not df.empty:
            # For simplicity, we'll treat each series as a document.
            # These are US-specific by default.
            country = "United States"
            content = f"Historical data for {series_name} in the {country}:\n\n"
            content += df.to_string(index=False)
            
            doc = {
                "document_id": str(uuid.uuid4()),
                "title": f"{series_name} - {country}",
                "source": "FRED API",
                "content": content,
                "publication_date": None,
                "country": country,
                "indicator": series_name,
            }
            all_documents.append(doc)
            
    return all_documents

if __name__ == '__main__':
    print("--- Loading FRED Data ---")
    # This will make live API calls if the key is set
    if not settings.FRED_API_KEY:
        print("Cannot run example: FRED_API_KEY is not set in your .env file.")
    else:
        fred_documents = load_fred_data()
        
        if fred_documents:
            print(f"\nSuccessfully loaded {len(fred_documents)} documents from the FRED API.")
            print("\n--- Example Document ---")
            example_doc = fred_documents[0]
            print(f"Title: {example_doc['title']}")
            print(f"Source: {example_doc['source']}")
            print(f"Country: {example_doc['country']}")
            print(f"Indicator: {example_doc['indicator']}")
            print("\n--- Content Snippet ---")
            print(example_doc['content'][:300] + "...")
        else:
            print("Failed to load any documents from the FRED API. Check your API key and network.")
