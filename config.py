import os
from dotenv import load_dotenv

# Read the secret keys from the .env file
load_dotenv()

# ---------- MODELS ----------
CHAT_MODEL = "gemini-3.8-flash"                # the brain that thinks and writes
EMBEDDING_MODEL = "models/gemini-embedding-001"  # turns text into numbers

# ---------- PDF CUTTING (chunking) ----------
CHUNK_SIZE = 1000       # each piece is about 1000 characters
CHUNK_OVERLAP = 200     # neighbouring pieces share 200 characters

# ---------- SEARCH ----------
TOP_K = 4               # how many pieces to fetch per search

# ---------- AGENT RULES ----------
MAX_RETRIES = 2         # search again at most 2 times (the safety stop)
GOOD_SCORE = 7          # evidence score (0-10) at or above this = "good"

# ---------- FOLDERS ----------
DOCUMENTS_FOLDER = "documents"   # where your PDFs live
INDEX_FOLDER = "faiss_index"     # where the saved search index goes

# ---------- SAFETY CHECK ----------
if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY is missing. Add it to your .env file.")