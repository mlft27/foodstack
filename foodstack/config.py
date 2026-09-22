import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your .env file or environment.")

# ---------- LangChain LLM ----------
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

# ---------- Embeddings (for RAG) ----------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")



