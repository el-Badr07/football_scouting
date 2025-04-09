import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
import glob # To find files in the data directory
# Added for chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document 

def load_api_keys():
    """Loads API keys from the .env file."""
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    # We might not need OpenAI key if solely using Groq + local embeddings
    # openai_api_key = os.getenv("OPENAI_API_KEY") 
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in .env file.")
    return groq_api_key

def get_embedding_model(model_name="all-MiniLM-L6-v2"):
    """Initializes and returns the Sentence Transformer embedding model."""
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    print("Embedding model loaded.")
    return model

def get_llm(api_key, model_name="llama3-8b-8192"):
    """Initializes and returns the Groq LLM."""
    print(f"Initializing Groq LLM: {model_name}...")
    llm = ChatGroq(temperature=0, groq_api_key=api_key, model_name=model_name)
    print("Groq LLM initialized.")
    return llm

def load_scouting_reports(data_folder=r"C:\Users\HP\Documents\football_scouting\data\data", chunk_size=1000, chunk_overlap=200):
    """
    Loads text content from .txt files, splits them into chunks, 
    and returns a list of Langchain Document objects.
    """
    raw_documents = []
    script_dir = os.path.dirname(__file__)
    absolute_data_folder = os.path.join(script_dir, data_folder)
    txt_files = glob.glob(os.path.join(absolute_data_folder, "*.txt"))

    if not txt_files:
        print(f"Warning: No .txt files found in '{absolute_data_folder}'. Using dummy data.")
        # Create dummy Document objects if no files found
        dummy_content = [
            "Player A is a fast winger with good dribbling skills but needs improvement in finishing. He excels in one-on-one situations and creating chances.",
            "Player B is a tall, strong central defender, excellent in aerial duels and tackling. His positioning is generally good, but he can be caught out by pace."
        ]
        for i, content in enumerate(dummy_content):
             # Add minimal metadata for consistency
            raw_documents.append(Document(page_content=content, metadata={"source": f"dummy_report_{i+1}.txt"})) 
    else:
        print(f"Loading reports from {len(txt_files)} files in '{absolute_data_folder}'...")
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Create a Document object for the whole file first
                    raw_documents.append(Document(page_content=content, metadata={"source": os.path.basename(file_path)}))
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
    
    if not raw_documents:
        print("No documents were loaded or created.")
        return []

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True, # Optional: adds character index of chunk start
    )
    
    print(f"Splitting {len(raw_documents)} documents into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    # Split the documents
    chunked_documents = text_splitter.split_documents(raw_documents)
    
    print(f"Created {len(chunked_documents)} chunks.")
    return chunked_documents 