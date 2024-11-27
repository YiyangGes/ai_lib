from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader
)
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
import os

groq_api = str(os.environ.get("GROQ_API_KEY"))

llm = Groq(model="llama3-groq-70b-8192-tool-use-preview", api_key=groq_api)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512
