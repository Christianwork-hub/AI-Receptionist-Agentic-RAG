"""
Configurazione dell'Architettura Agentic RAG
---------------------------------------------
Questo modulo centralizza la configurazione per l'intero sistema: 
integrazione LLM, embedding model (dense e sparse), database vettoriale (Qdrant) 
e i parametri di ingestion e chunking per l'ottimizzazione del recupero documentale.
"""
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import FastEmbedSparse
from qdrant_client import QdrantClient

load_dotenv()

DB_URI = os.getenv("DATABASE_URI")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


# --- Integrazione LLM (Reasoning Engine) ---
# Modello DeepSeek utilizzato per le funzionalità di planning, reasoning e orchestrazione dei tool.
# NOTA: Disabilitare esplicitamente il "thinking mode" per evitare latenza e formati di output
# non compatibili con i protocolli di tool calling standard di LangChain.
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url="https://api.deepseek.com",
    api_key=DEEPSEEK_API_KEY,
    temperature=0,  # Temperatura 0 per massimizzare la coerenza logica e la precisione del tool calling
    model_kwargs={
        # Disabilitazione thinking mode per compatibilità con il tool calling framework
    },
)

# --- Configurazione alternativa con modello locale Ollama (commentata) ---
# Per usare questa configurazione, commenta il blocco 'llm' qui sopra e rimuovi
# i commenti (#) dal blocco qui sotto. Ricordati di importare ChatOllama se non presente.
# (es. from langchain_ollama import ChatOllama)

# OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
# LLM_MODEL = "ollama/gpt-oss:120b" #scegli il tuo modello Ollama  (es. "ollama/qwen 3.5:9b" o "ollama/llama3.1:8b)
# LLM_API_BASE = "https://ollama.com" #nel caso di gpt:oss:120 altrimenti https://localhost:11434 se usi un modello ollama locale

# llm = ChatOllama(
#     model=LLM_MODEL,
#     base_url=LLM_API_BASE,
#     temperature=0
# )







# --- Modelli di Embedding (Dense & Sparse) ---
# Integrazione di bge-m3 tramite Ollama per generare i dense vector (ricerca semantica).
dense_embeddings = OllamaEmbeddings(
    model="bge-m3:latest",
    base_url="http://localhost:11434",
)

# Aggiunta di FastEmbedSparse (BM25) per ricerca esatta su keyword (lexical search),
# combinato al dense vector per una ricerca vettoriale ibrida ad alta efficienza.
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

# --- Configurazione Database Vettoriale (Qdrant) ---
COLLECTION_HOTEL = "conoscenza_hotel"
COLLECTION_WEB = "conoscenza_web"
SPARSE_VECTOR_NAME = "sparse"
EMBEDDING_DIM = 1024

qdrant_client = QdrantClient(path="./qdrant_data")

# --- Path store ---
PARENT_STORE_PATH = "./parent_store"
MARKDOWN_DIR = "./markdown_docs"

# --- Configurazione Chunking Semantico ---
# Strategia "Parent-Child" per ottimizzare il bilanciamento tra la precisione del recupero 
# (child chunks piccoli e densi) e il mantenimento del contesto semantico (parent chunks espansi).

# Child Chunks: Segmenti ad alta granularità indicizzati nel Vector Store per il retrieval di precisione.
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 100

# Parent Chunks: Unità semantiche più estese archiviate su file system, per fornire il contesto macro all'LLM.
MIN_PARENT_SIZE = 2000
MAX_PARENT_SIZE = 4000
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
]

# --- Sicurezza e Limiti di Esecuzione Orchestratore ---
# Threshold di sicurezza per prevenire tool loop infiniti in query complesse o ambigue.
MAX_TOOL_CALLS = 8
MAX_ITERATIONS = 10

# --- Gestione del Context Window (Memory Management) ---
# Limite operativo di token. Se superato, si innesca la "context compression" per riassumere lo stato.
BASE_TOKEN_THRESHOLD = 2000
TOKEN_GROWTH_FACTOR = 0.9

