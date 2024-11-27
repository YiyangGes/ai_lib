from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader
)
from llama_index.llms.groq import Groq
from dotenv import load_dotenv
import os

groq_api = str(os.environ.get("GROQ_API_KEY"))

llm = Groq(model="llama3-groq-70b-8192-tool-use-preview", api_key=groq_api)
