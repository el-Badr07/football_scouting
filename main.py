import faiss
import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
# Import Document for type hinting if needed, though not strictly required here
from langchain_core.documents import Document 

# Assuming utils.py is in the same directory or Python path
from utils import (
    load_api_keys, 
    get_embedding_model, 
    get_llm, 
    load_scouting_reports
)

def create_vector_store(chunked_documents, embedding_model):
    """Creates a FAISS vector store from document chunks and an embedding model."""
    # Extract page content from Document objects
    contents = [doc.page_content for doc in chunked_documents]
    if not contents:
        print("No document chunk content found to create vector store.")
        return None, None
        
    print(f"Creating embeddings for {len(contents)} document chunks...")
    embeddings = embedding_model.encode(contents)
    embeddings = np.array(embeddings).astype('float32')
    
    dimension = embeddings.shape[1]
    print(f"Building FAISS index with dimension {dimension}...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"FAISS index built successfully with {index.ntotal} vectors.")
    
    # Store the original Document objects (containing content and metadata)
    return index, chunked_documents 

def format_docs(docs_with_metadata):
    """Helper function to format retrieved document chunks including source metadata."""
    formatted_docs = []
    for i, doc in enumerate(docs_with_metadata):
        # doc is expected to be a Document object now
        source = doc.metadata.get('source', 'Unknown Source')
        formatted_docs.append(f"Excerpt from {source} (Chunk {i+1}):\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted_docs)

def main():
    # --- Initialization ---
    try:
        groq_api_key = load_api_keys()
        embedding_model = get_embedding_model()
        llm = get_llm(groq_api_key)
        # Now loads chunked Document objects
        chunked_documents = load_scouting_reports(chunk_size=500, chunk_overlap=100) # Smaller chunks
        
        if not chunked_documents:
            print("No scouting report chunks loaded. Exiting.")
            return

        # Pass the chunked documents (list of Document objects)
        vector_store, stored_documents = create_vector_store(chunked_documents, embedding_model)
        
        if vector_store is None:
            print("Failed to create vector store. Exiting.")
            return

    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    # --- RAG Setup ---
    # Retriever function now works with stored Document objects
    def retrieve_documents(query, k=3):
        print(f"\nSearching for document chunks related to: '{query}'")
        query_embedding = embedding_model.encode([query])[0].astype('float32').reshape(1, -1)
        distances, indices = vector_store.search(query_embedding, k)
        
        # Retrieve the full Document objects using the indices
        retrieved_docs_with_metadata = [stored_documents[i] for i in indices[0]]
        
        print(f"Retrieved {len(retrieved_docs_with_metadata)} document chunks.")
        return retrieved_docs_with_metadata # Return list of Document objects

    # Define the prompt template (remains the same, but context will be richer)
    template = """You are a professional football scout. Based on the following scouting report excerpts (including their source file), answer the user's question comprehensively. Provide details about potential players matching the criteria, mentioning their strengths and weaknesses as described in the reports. If the context doesn't provide a clear answer, say so.

Context:
{context}

Question: {question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    # Define the RAG chain (logic remains the same, relies on updated functions)
    rag_chain = (
        {"context": (lambda x: format_docs(retrieve_documents(x["question"]))), 
         "question": lambda x: x["question"] 
         }
        | prompt
        | llm
        | StrOutputParser()
    )

    # --- Interaction Loop (remains the same) ---
    print("\n--- Football Scout Assistant (with Chunking) ---")
    print("Enter your query (e.g., 'Find me a fast winger with good dribbling') or type 'quit' to exit.")
    
    while True:
        user_query = input("\nYour Query: ")
        if user_query.lower() == 'quit':
            break
        if not user_query:
            continue
            
        print("Processing your query...")
        try:
            result = rag_chain.invoke({"question": user_query})
            print("\n--- Scout's Analysis ---")
            print(result)
            print("------------------------")
        except Exception as e:
            print(f"An error occurred: {e}")

    print("Exiting assistant. Goodbye!")

if __name__ == "__main__":
    main()
