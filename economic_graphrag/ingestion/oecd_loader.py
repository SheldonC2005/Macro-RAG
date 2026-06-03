# economic_graphrag/ingestion/oecd_loader.py
import requests
import pandas as pd
from typing import List, Dict, Any
import uuid

# G7 countries for OECD data to keep the scope manageable
G7_COUNTRIES = {
    "CAN": "Canada", "FRA": "France", "DEU": "Germany", "ITA": "Italy",
    "JPN": "Japan", "GBR": "United Kingdom", "USA": "United States"
}

# A key OECD indicator: Quarterly GDP growth
OECD_INDICATOR = "QNA" # Quarterly National Accounts
MEASURE = "GPSA" # Gross Domestic Product, expenditure approach, seasonally adjusted
SUBJECT = "B1_GE" # Gross domestic product - expenditure approach
FREQUENCY = "Q" # Quarterly

def get_oecd_data(country_code: str, start_year: int = 2000) -> pd.DataFrame:
    """
    Fetches quarterly GDP data for a specific country from the OECD SDMX-JSON API.
    """
    # The OECD API structure is complex. This is a targeted query for one indicator.
    # URL format: /<database>/<dataset>/<filter>/<agency>/<version>
    # Example filter: DEU.B1_GE.GPSA.Q
    dataset_filter = f"{country_code}.{SUBJECT}.{MEASURE}.{FREQUENCY}"
    url = f"https://stats.oecd.org/SDMX-JSON/data/{OECD_INDICATOR}/{dataset_filter}/all"
    params = {
        "startTime": f"{start_year}-Q1",
        "dimensionAtObservation": "AllDimensions"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if "dataSets" in data and data["dataSets"]:
            obs = data["dataSets"][0]["observations"]
            series = data["series"]
            
            records = []
            for key, value in obs.items():
                # Key is like "0:0:0:0:0", mapping to dimensions
                obs_index = int(key.split(':')[0])
                time_period = series['0:0:0:0']['attributes'][1]['values'][obs_index]['name']
                gdp_value = value[0]
                
                records.append({
                    "date": time_period,
                    "value": gdp_value
                })
            
            df = pd.DataFrame(records)
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            return df.dropna()

    except requests.exceptions.RequestException as e:
        print(f"Error fetching OECD data for {country_code}: {e}")
    except (KeyError, IndexError) as e:
        print(f"Error parsing OECD response for {country_code}: {e}")
        
    return pd.DataFrame()

def load_oecd_data() -> List[Dict[str, Any]]:
    """
    Loads quarterly GDP data from the OECD for G7 countries and formats it
    into the document structure for the RAG system.
    """
    all_documents = []
    indicator_name = "Quarterly GDP (expenditure approach, seasonally adjusted)"

    for country_code, country_name in G7_COUNTRIES.items():
        print(f"Fetching Quarterly GDP for {country_name} from OECD...")
        df = get_oecd_data(country_code)
        
        if not df.empty:
            content = f"Historical data for {indicator_name} in {country_name}:\n\n"
            content += df.to_string(index=False)
            
            doc = {
                "document_id": str(uuid.uuid4()),
                "title": f"{indicator_name} - {country_name}",
                "source": "OECD API",
                "content": content,
                "publication_date": None,
                "country": country_name,
                "indicator": indicator_name,
            }
            all_documents.append(doc)
            
    return all_documents

if __name__ == '__main__':
    print("--- Loading OECD Data ---")
    # This will make live API calls
    oecd_documents = load_oecd_data()
    
    if oecd_documents:
        print(f"\nSuccessfully loaded {len(oecd_documents)} documents from the OECD API.")
        print("\n--- Example Document ---")
        example_doc = oecd_documents[0]
        print(f"Title: {example_doc['title']}")
        print(f"Source: {example_doc['source']}")
        print(f"Country: {example_doc['country']}")
        print(f"Indicator: {example_doc['indicator']}")
        print("\n--- Content Snippet ---")
        print(example_doc['content'][:300] + "...")
    else:
        print("Failed to load any documents from the OECD API.")
