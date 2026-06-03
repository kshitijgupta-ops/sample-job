from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
from logger import get_logger

logger = get_logger(__name__)

if os.environ.get("ENV", "staging") == "dev":
    # local paths for development, resolved relative to this script
    path = os.path.join(os.path.dirname(__file__), "Indian_Premier_League.pdf")
    vector_store_path = os.path.join(os.path.dirname(__file__), "vector-store")
else:
    path = "/mnt/rag/Indian_Premier_League.pdf"
    vector_store_path = "/mnt/rag/vector-store"

print("Using document path:", path)  # Debugging statement to confirm path
try:
    # Load documents from mounted volume
    loader = PyPDFLoader(path)
    docs = loader.load()
except Exception as e:
    logger.error("Document loading failed: %s", str(e))
    raise

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Your embedding code (API key pulled from env variable for security)
try:
    embeddings = OpenAIEmbeddings(
        model=os.environ.get("EMBEDDING_MODEL", "openai-main/text-embedding-3-small"),
        base_url=os.environ["EMBEDDING_BASE_URL"],   # full Gateway Base URL (e.g. https://<host>/api/llm)
        api_key=os.environ["EMBEDDING_API_KEY"],      # store as env var
        default_headers={
            "X-TFY-METADATA": "{}",
            "X-TFY-LOGGING-CONFIG": '{"enabled": true}',
        },
        # Gateway handles tokenization; don't let langchain pre-tokenize into token IDs
        check_embedding_ctx_length=False,
    )
except Exception as e:
    logger.error("Embedding initialization failed: %s", str(e))
    raise

# Create vector store and save to mounted volume
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local(vector_store_path)
logger.info("Vector store saved successfully!")