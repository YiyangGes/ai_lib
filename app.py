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
from datetime import datetime
import json
from llama_index.llms.groq import Groq

# from dotenv import load_dotenv
# import os

groq_api = str(os.environ.get("GROQ_API_KEY"))

# llm = Groq(model="llama3-groq-70b-8192-tool-use-preview", api_key=groq_api, request_timeout = 500)
# phi3.5:3.8b-mini-instruct-q8_0
llm = Ollama(model='phi3.5:latest ', request_timeout=500)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 1024

tools = []

def list_to_string(lst):
    if not lst:
        return ""
    elif len(lst) == 1:
        return lst[0]
    else:
        return ", ".join(lst[:-1]) + " and " + lst[-1]


def list_uploaded_files(): # list all files uploaded
    UPLOAD_DIR = "upload"
    if not os.path.exists(UPLOAD_DIR):
        return []
    return [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]


# Function to process uploaded file(s) and create a vector database
def create_vector_database(file_input):
    # global vector_database
    # Directory to store uploaded files
    UPLOAD_DIR = "upload"
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    if not file_input:
        return "No files uploaded."

    shutil.copy(file_input, UPLOAD_DIR)

    vectDB = os.path.join("DB", os.path.basename(file_input.name))
    file_p = os.path.join(UPLOAD_DIR, os.path.basename(file_input.name))

    if os.path.exists(vectDB):
        return vectDB
    
    # # Read files using SimpleDirectoryReader
    documents = SimpleDirectoryReader(input_files=[file_p]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=vectDB)
    # file_name = file_input.name[file_input.name.rfind("\\")+1:]
    # vector_list[file_name] = vectDB
    # new_options.append(file_name)
    # print(vector_list)

    return vectDB


def history_vdb(file_path):
    if os.path.exists("DB/"+file_path):
        return "Already exist"
    else:
        documents = SimpleDirectoryReader(input_files=["upload/"+file_path]).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir="DB/"+file_path)


def upload_file(files):
    print("here")
    file_paths = [file.name for file in files]
    return file_paths


def GetSources(response):
    pages = []
    sources = []
    
    for node in response.source_nodes:
      # Access the TextNode object directly
      text_node = node.node

      # Assuming metadata is stored within the TextNode's metadata
      source = text_node.metadata.get('file_name') # Access metadata using .metadata.get()
      page = text_node.metadata.get('page_label')  # Access metadata using .metadata.get()
      sources.append(source)
      pages.append(page)
    #   print(f"Source: {source} \nPage:{page}")

    return f"\nSource: {list_to_string(list(set(sources)))} \nPage: {list_to_string(list(set(pages)))}"


def respond(message, chat_history, vdb='DB/theory_never_die.pdf'):

    uploaded_files = list_uploaded_files()
    uploaded_files_dict = {file: f"DB/{file}" for file in uploaded_files}

    storage_context = StorageContext.from_defaults(persist_dir=vdb)
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine(similarity_top_k=5)

    response = query_engine.query(message)

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": str(response)+ '\n'+ GetSources(response)})
    return "", chat_history, vdb


def download_his(chat_history,_):
    # Specify the file name
    # file_name = "conversation.txt"
    # Get the current date and time
    print(chat_history)
    now = datetime.now()

    # Format it as a string for the file name
    file_name = now.strftime("%Y-%m-%d_%H-%M-%S.json")

    # Write the list to a text file
    with open('upload/'+file_name, "w") as file:
        # Save the list to a JSON file
        json.dump(chat_history, file, indent=4)
        # for entry in chat_history:
        #     file.write(f"{entry['role']}: {entry['content']}\n")
    history_vdb(file_name)
    
    return chat_history, file_name+ " downloaded at upload/" + file_name


# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Vector Database Creation and Selection")

    # # Step 1: Upload file(s)
    file_input = gr.File(label="Upload your Ebook", file_types=[".pdf", ".epub"], file_count = "single")
    # print(file_input)

    # # # Step 2: Create vector database
    create_db_button = gr.Button("Process the Book")
    db_creation_output = gr.Textbox('DB/theory_never_die.pdf', label="Database Created at", interactive=False)
    # drop_down = gr.Dropdown(loaded_files, label = "Document", info = "Choose the document you would like to have a conversation with")

    create_db_button.click(
        create_vector_database,
        inputs=file_input,
        outputs=db_creation_output
        )

    # db_creation_output.change(update_dropdown, [db_creation_output, drop_down],[db_creation_output, drop_down])

    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(placeholder="Ask me informations about given document")
    gr.Examples(
        examples=[
            "Tell me a fun story in the book",
            "What is one of the most important thing Bayes did",
            "Tell me about 5 Applications of Bayes Theorem"
        ],
        inputs=msg
    )
    button = gr.Button(value="Submit")

    msg.submit(respond, [msg, chatbot, db_creation_output], [msg, chatbot, db_creation_output])
    button.click(respond, [msg, chatbot, db_creation_output], [msg, chatbot, db_creation_output])

    history_status = gr.Textbox(label = "Status of history downloading", interactive=False)
    button_his = gr.Button(value = "Download History")
    button_his.click(download_his, [chatbot,history_status], [chatbot, history_status])

    clear_button = gr.Button("Clear Chat")
    clear_button.click(lambda: [], inputs=None, outputs=chatbot)  # Clear the chat history

# Launch the app
demo.launch()
