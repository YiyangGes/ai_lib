import os
import shutil
import json
from datetime import datetime
import gradio as gr

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
from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI


# API Key for Groq model
GROQ_API_KEY = str(os.environ.get("GROQ_API_KEY"))
OPENAI_API_KEY = str(os.environ.get("OPENAI_API_KEY"))


# llm = OpenAI(
#     model="gpt-4o-mini",
#     api_key=OPENAI_API_KEY,  # uses OPENAI_API_KEY env var by default
# )

# Define LLM and embedding model
# llm = Groq(model="llama3-groq-70b-8192-tool-use-preview", api_key=GROQ_API_KEY, request_timeout = 500)
llm = Ollama(model='phi3.5:latest', request_timeout=500)  # Change model as needed, very slow
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Configure settings for LLM and embeddings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 1024

# Initialize tools list for future extensions
# tools = []

# Utility Functions
def list_databases():
    """
    Lists all databases in the 'DB' directory.
    """
    DB_DIR = "DB"
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
    return [os.path.join(DB_DIR, f) for f in os.listdir(DB_DIR) if os.path.isdir(os.path.join(DB_DIR, f))]

def list_to_string(lst):
    """
    Converts a list into a string with proper grammar for joining elements.
    """
    if not lst:
        return ""
    elif len(lst) == 1:
        return lst[0]
    else:
        return ", ".join(lst[:-1]) + " and " + lst[-1]

def list_uploaded_files():
    """
    Lists all files in the 'upload' directory.
    """
    UPLOAD_DIR = "upload"
    if not os.path.exists(UPLOAD_DIR):
        return []
    return [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]

# Vector Database Functions

def create_vector_database(file_input):
    """
    Processes an uploaded file and creates a vector database.
    """
    UPLOAD_DIR = "upload"
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    if not file_input:
        return "No files uploaded."

    shutil.copy(file_input, UPLOAD_DIR)

    vectDB = os.path.join("DB", os.path.basename(file_input.name))
    file_p = os.path.join(UPLOAD_DIR, os.path.basename(file_input.name))

    if os.path.exists(vectDB):
        return vectDB

    # Read files using SimpleDirectoryReader
    documents = SimpleDirectoryReader(input_files=[file_p]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=vectDB)

    return vectDB

def history_vdb(file_path):
    """
    Adds file to vector database if not already existing.
    """
    if os.path.exists("DB/" + file_path):
        return "Already exists"
    else:
        documents = SimpleDirectoryReader(input_files=["upload/" + file_path]).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir="DB/" + file_path)

def upload_file(files):
    """
    Handles file uploads and returns a list of file paths.
    """
    return [file.name for file in files]

# Helper Functions

def GetSources(response):
    """
    Extracts and formats sources and pages from the response metadata.
    """
    pages = []
    sources = []

    for node in response.source_nodes:
        text_node = node.node
        source = text_node.metadata.get('file_name')
        page = text_node.metadata.get('page_label')
        sources.append(source)
        pages.append(page)

    return f"\nSource: {list_to_string(list(set(sources)))} \nPage: {list_to_string(list(set(pages)))}"

def respond(message, chat_history, vdb='DB/theory_never_die.pdf'):
    """
    Handles user queries and updates the chat history with responses and sources.
    """
    uploaded_files = list_uploaded_files()
    uploaded_files_dict = {file: f"DB/{file}" for file in uploaded_files}

    storage_context = StorageContext.from_defaults(persist_dir=vdb)
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine(similarity_top_k=5)

    response = query_engine.query(message)

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": str(response) + '\n' + GetSources(response)})
    return "", chat_history, vdb

def download_his(chat_history, _):
    """
    Downloads the chat history to a JSON file and updates the vector database.
    """
    now = datetime.now()
    file_name = now.strftime("%Y-%m-%d_%H-%M-%S.json")

    with open('upload/' + file_name, "w") as file:
        json.dump(chat_history, file, indent=4)

    history_vdb(file_name)
    return chat_history, file_name + " downloaded at upload/" + file_name

# Gradio Interface
with gr.Blocks() as demo:
    # The Gradio interface serves as the user-friendly frontend for the application, enabling users to upload documents, create vector databases, and interact with the chatbot for querying document content.
    gr.HTML("<h1 style='text-align: center;'>Hello, I am <strong>AI Librarian</strong>!</h1>")  # Centered title
    gr.HTML("<h2 style='text-align: center;'>Start to chat with me for any informations about the books!</h2>")  # Centered title        
    gr.Markdown('---')
    gr.Markdown("## Vector Database Creation and Selection")
    gr.Markdown('1. Upload your book or use the defalt book theory_never_die.pdf\n '+
                '2. Click the Process the Book button to process the uploaded book.(that would be stored at /upload dir\n'+
                '3. When the database is created, Start the chat :)')
    
    
    file_input = gr.File(label="Upload your Ebook", file_types=[".pdf", ".epub"], file_count="single")  # Allows users to upload an ebook in supported formats to create a vector database.
    create_db_button = gr.Button("Process the Book")
    db_creation_output = gr.Textbox('DB/theory_never_die.pdf', label="Database Created at (File path of the generated vector database)", interactive=False)

    db_example = gr.Examples(
        examples=list_databases(),
        inputs=db_creation_output,
        label="Select a Database",
    )
    gr.Markdown("Retart to update the options after upload a book :( ")
    
    # db_creation_output.change(update_examples, db_example, db_example)

    create_db_button.click(
        create_vector_database,
        inputs=file_input,
        outputs=db_creation_output
    )

    gr.Markdown('---')

    gr.Markdown("## Chat Section")

    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(placeholder="Ask me information about the given document")
    gr.Examples(
        examples=[
            "Tell me a fun story in the book",
            "What is one of the most important things Bayes did",
            "Tell me about 5 Applications of Bayes Theorem"
        ],
        inputs=msg
    )  # These examples provide predefined queries to help users understand the types of questions they can ask.
    button = gr.Button(value="Submit")

    msg.submit(respond, [msg, chatbot, db_creation_output], [msg, chatbot, db_creation_output])
    button.click(respond, [msg, chatbot, db_creation_output], [msg, chatbot, db_creation_output])

    gr.Markdown('---')

    gr.Markdown("## Chat History Download Section")

    history_status = gr.Textbox(label="Status of history downloading", interactive=False)
    button_his = gr.Button(value="Download History")
    button_his.click(download_his, [chatbot, history_status], [chatbot, history_status])

    clear_button = gr.Button("Clear Chat")
    # Note: Clearing the chat resets the  chat history. It does not affect the session state or database.
    clear_button.click(lambda: [], inputs=None, outputs=chatbot)

# Launch the Gradio app
demo.launch()
