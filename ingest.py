from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
import os

from logger import get_logger

logger = get_logger(__name__)

try:
    # Load documents from mounted volume
    loader = PyPDFLoader("/mnt/rag/Indian_Premier_League.pdf")
    docs = loader.load()
except Exception as e:
    logger.error("Document loading failed: %s", str(e))
    raise

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Your embedding code (API key pulled from env variable for security)
try:
    embeddings = OpenAIEmbeddings(
        model="truefoundry/truefoundry",
        base_url=os.environ["EMBEDDING_BASE_URL"],   # store as env var
        api_key=os.environ["EMBEDDING_API_KEY"],      # store as env var
        extra_headers={
            "X-TFY-METADATA": '{}',
            "X-TFY-LOGGING-CONFIG": '{"enabled": true}'
        }
    )
except Exception as e:
    logger.error("Embedding initialization failed: %s", str(e))
    raise

# Create vector store and save to mounted volume
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("/mnt/vector-store")
logger.info("Vector store saved successfully!")