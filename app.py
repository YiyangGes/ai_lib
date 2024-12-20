import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    load_index_from_storage,
    Settings
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
import gradio as gr
import shutil

# from llama_index.llms.groq import Groq
# from dotenv import load_dotenv

# 2. Document Processing
# Use LlamaIndex's SimpleDirectoryReader
# Support multiple ebook formats (.epub, .pdf)
# Implement robust text extraction
# Chunk documents for efficient embedding

# 3. Embedding Generation
# Use lightweight embedding model (e.g., BAAI/bge-small-en-v1.5)
# Create vector index for semantic search
# Implement efficient indexing strategy

# 4. Query Engine
# Configure Phi3.5 from Ollama
# Implement RAG pipeline
# Force citation generation
# Handle conversation history

# 5. Gradio Interface Design
# Create upload/import functionality
# Design query interface with:
# Search input
# Conversation history display
# Source citation display
# Premade prompt examples

# 6. Additional Features
# Save/export conversation history
# Add documents to library from conversation
# Error handling for document processing

# client = OpenAI(
# base_url='http://localhost:11434/v1/',
# api_key='ollama' # Required but ignored by Ollama
# # )
llm = Ollama(model='phi3.5:3.8b-mini-instruct-q8_0', request_timeout=300)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512

# docs_bayes = SimpleDirectoryReader(input_files=["books/theory_never_die.pdf"]).load_data()

# vectDB = "DB/DB_txtb1"
# index = VectorStoreIndex.from_documents(docs_bayes)
# query_engine = index.as_query_engine(similarity_top_k=5)
# index.storage_context.persist(persist_dir=vectDB)

# storage_context = StorageContext.from_defaults(persist_dir=vectDB)

# index = load_index_from_storage(storage_context)
# query_engine = index.as_query_engine(similarity_top_k=5)

# response = query_engine.query("Tell me about 5 Applications of Bayes Theorem")
# print("Storage Context:", storage_context)
# print("Index Loaded:", index)
# print("Query Engine Initialized:", query_engine)

# response = query_engine.query("What is the name of part I of this book?")
# print(str(response))
# indexing strategy,
# 1. upload
# 2. process, save vector db 
# 3. load index from storage

# conversation history,
# save conversasion history as a document

# Gradio
vector_list = []

# Function to process uploaded file(s) and create a vector database
def create_vector_database(file_input):
    # global vector_database
    # Directory to store uploaded files
    UPLOAD_DIR = "upload"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    shutil.copy(file_input, UPLOAD_DIR)

    if not file_input:
        return "No files uploaded."

    vectDB = os.path.join("DB", os.path.basename(file_input.name))
    file_p = os.path.join(UPLOAD_DIR, os.path.basename(file_input.name))

    # # Read files using SimpleDirectoryReader
    documents = SimpleDirectoryReader(input_files=[file_p]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=vectDB)

    vector_list.append(vectDB)

    return "Vector database created successfully!"


# def query(vdb):
#     # storage_context = StorageContext.from_defaults(persist_dir=vectDB)

# # index = load_index_from_storage(storage_context)
# # query_engine = index.as_query_engine(similarity_top_k=5)

def upload_file(files):
    print("here")
    file_paths = [file.name for file in files]
    return file_paths

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Vector Database Creation and Selection")

    # # Step 1: Upload file(s)
    file_input = gr.File(label="Upload your Ebook", file_types=[".pdf"], file_count = "single")
    # print(file_input)

    # # # Step 2: Create vector database
    create_db_button = gr.Button("Process the Book")
    db_creation_output = gr.Textbox(label="Database Creation Status", interactive=False)
    # Link components
    create_db_button.click(
        create_vector_database,
        inputs=file_input,
        outputs=db_creation_output
    )
    # # Step 3: Select vector database
    # select_db_button = gr.Button("Select Vector Database for RAG")
    # db_selection_output = gr.Textbox(label="Database Selection Status", interactive=False)


    # select_db_button.click(
    #     select_vector_database,
    #     inputs=None,
    #     outputs=db_selection_output
    # )

# Launch the app
demo.launch()
