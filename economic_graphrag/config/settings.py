# economic_graphrag/config/settings.py
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
FRED_API_KEY = os.getenv("FRED_API_KEY")

# --- Project Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
WORLDBANK_DIR = DATA_DIR / "worldbank"
FRED_DIR = DATA_DIR / "fred"
OECD_DIR = DATA_DIR / "oecd"
TESTS_DIR = BASE_DIR / "tests"

# --- Gemini Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# --- ChromaDB Configuration ---
CHROMA_PATH = os.getenv("CHROMA_PATH", str(DATA_DIR / "chroma_db"))
CHROMA_COLLECTION_NAME = "macro_economics"

# --- Neo4j Configuration ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# --- Document Processing Configuration ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# --- Hybrid Retrieval Configuration ---
GRAPH_WEIGHT = 0.4
VECTOR_WEIGHT = 0.6

# --- Logging Configuration ---
LOGGING_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
