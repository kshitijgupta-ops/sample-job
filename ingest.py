from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import os
from dotenv import load_dotenv
load_dotenv()
from logger import get_logger
logger = get_logger(__name__)

# ─────────────────────────────────────
# PATHS
# ─────────────────────────────────────
if os.environ.get("ENV", "staging") == "dev":
    # local paths for development, resolved relative to this script
    path = os.path.join(os.path.dirname(__file__), "Indian_Premier_League.pdf")
else:
    path = "/mnt/rag/Indian_Premier_League.pdf"

# Qdrant config from env vars
QDRANT_HOST = os.environ.get("QDRANT_HOST", "ml.tfy-eo.truefoundry.cloud")
QDRANT_PATH = os.environ.get("QDRANT_PATH", "qdrant-vectordb-kshitij-test")
QDRANT_URL        = "https://"+QDRANT_HOST+"/"+QDRANT_PATH
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "ipl-docs")

logger.info("Using document path: %s", path)

# ─────────────────────────────────────
# STEP 1: Load PDF
# ─────────────────────────────────────
try:
    loader = PyPDFLoader(path)
    docs = loader.load()
    logger.info("Loaded %d pages from PDF", len(docs))
except Exception as e:
    logger.error("Document loading failed: %s", str(e))
    raise

# ─────────────────────────────────────
# STEP 2: Split into chunks
# ─────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
chunks = splitter.split_documents(docs)
logger.info("Split into %d chunks", len(chunks))

# ─────────────────────────────────────
# STEP 3: Embeddings (unchanged)
# ─────────────────────────────────────
try:
    embeddings = OpenAIEmbeddings(
        model=os.environ.get("EMBEDDING_MODEL", "openai-main/text-embedding-3-small"),
        base_url=os.environ["BASE_URL"],    # full Gateway Base URL
        api_key=os.environ["EMBEDDING_API_KEY"],       # store as env var
        default_headers={
            "X-TFY-METADATA": "{}",
            "X-TFY-LOGGING-CONFIG": '{"enabled": true}',
        },
        # Gateway handles tokenization; don't let langchain pre-tokenize
        check_embedding_ctx_length=False,
    )
except Exception as e:
    logger.error("Embedding initialization failed: %s", str(e))
    raise

# ─────────────────────────────────────
# STEP 4: Connect to Qdrant + create collection if needed
# ─────────────────────────────────────
client = None
try:
    client = QdrantClient(
        host=QDRANT_HOST,
        port=443,
        https=True,
        prefix=QDRANT_PATH,  # ← path prefix passed separately
        prefer_grpc=False,
        check_compatibility=False,
        timeout=30,
    )

    existing = [c.name for c in client.get_collections().collections]

    if QDRANT_COLLECTION not in existing:
        client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=1536,              # text-embedding-3-small output dimension
                distance=Distance.COSINE
            )
        )
        logger.info("Collection created: %s", QDRANT_COLLECTION)
    else:
        logger.info("Collection already exists: %s", QDRANT_COLLECTION)

except Exception as e:
    logger.error("Qdrant connection failed: %s", str(e))
    raise

# ─────────────────────────────────────
# STEP 5: Embed chunks and store in Qdrant
# ─────────────────────────────────────
try:
    # Initialize vectorstore pointing at existing collection
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings,
    )

    # Add documents (embeds + upserts in batches)
    vectorstore.add_documents(chunks)

    logger.info("Vector store saved to Qdrant successfully! %d chunks ingested.", len(chunks))
except Exception as e:
    logger.error("Qdrant upsert failed: %s", str(e))
    raise e