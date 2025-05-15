import ollama
import os
import numpy as np
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# --- Configuration ---
DOCUMENT_DIR = "data"  # Directory where your text files are stored
CHROMA_PERSIST_DIR = "chroma_db_ollama" # Directory to store ChromaDB data
COLLECTION_NAME = "rag_collection"    # Name of the collection in ChromaDB

EMBEDDING_MODEL_NAME = 'mxbai-embed-large' # The model you tested for embeddings
LLM_MODEL_NAME = 'gemma3:1b-it-qat'      # The LLM you tested for chat

# Initialize Ollama Embeddings
# This will be used by LangChain components
ollama_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)

# --- Helper Function for Cosine Similarity (can be removed if Chroma handles all scoring) ---
def cosine_similarity(v1, v2):
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

# --- Phase 1: Building or Loading the Knowledge Base ---
def load_or_create_knowledge_base():
    """
    Loads an existing ChromaDB vector store or creates a new one
    by loading, chunking, and embedding documents.
    Specifies 'cosine' similarity when creating a new store.
    """
    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR): # Check if directory exists and is not empty
        print(f"Loading existing vector store from: {CHROMA_PERSIST_DIR}")
        # When loading, the metric is already set from creation time.
        # Ensure you also update the Chroma import to: from langchain_chroma import Chroma
        vector_store = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=ollama_embeddings, # ollama_embeddings should be defined globally or passed
            collection_name=COLLECTION_NAME
        )
        # You can try to verify the metric of the loaded collection if needed, though it's not straightforward with just the LangChain wrapper.
        # For now, trust that it was created with the correct metric if the creation code path was hit previously with that setting.
        print(f"Vector store loaded. Collection '{COLLECTION_NAME}' has {vector_store._collection.count()} documents.")
    else:
        print("Creating new vector store with COSINE similarity...") # Indicate which metric is being used
        print(f"Loading documents from: {DOCUMENT_DIR}")
        # Using DirectoryLoader to load all .txt files
        loader_kwargs = {'encoding': 'utf-8'}
        loader = DirectoryLoader(
            DOCUMENT_DIR,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs=loader_kwargs, # Pass the encoding argument here
            show_progress=True,
            use_multithreading=True # Can speed up loading multiple files
        )
        print("Attempting to load documents with UTF-8 encoding...")
        try:
            documents = loader.load()
        except Exception as e:
            print(f"An error occurred during document loading: {e}")
            print("Please ensure all .txt files in the 'data' directory are UTF-8 encoded or try a different encoding if certain.")
            return None # Stop if loading fails

        if not documents:
            print("No documents found. Cannot build knowledge base.")
            return None

        print(f"Loaded {len(documents)} document(s).")

        print("Splitting documents into chunks...")
        # Using RecursiveCharacterTextSplitter for more sophisticated chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,  # Max characters per chunk
            chunk_overlap=150, # Characters of overlap between chunks
            length_function=len
        )
        doc_chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(doc_chunks)} chunks.")

        if not doc_chunks:
            print("No chunks created. Cannot build knowledge base.")
            return None

        print(f"Creating vector store and generating embeddings using model: {EMBEDDING_MODEL_NAME}")
        # Chroma will handle embedding generation using the provided ollama_embeddings
        # Specify the distance metric (similarity function) during creation
        vector_store = Chroma.from_documents(
            documents=doc_chunks,
            embedding=ollama_embeddings, # ollama_embeddings should be defined globally or passed
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_metadata={"hnsw:space": "cosine"}  # <--- ADD THIS LINE
        )
        print(f"Vector store created with COSINE similarity and persisted at: {CHROMA_PERSIST_DIR}")
        print(f"Collection '{COLLECTION_NAME}' has {vector_store._collection.count()} documents.")

    return vector_store

# --- Phase 2: Retrieval and Generation ---
def find_relevant_chunks_from_vectorstore(vector_store, query_text, top_n=2):
    if not vector_store:
        print("Vector store is not initialized.")
        return []
    print(f"\nSearching for relevant chunks for query: '{query_text[:50]}...'")

    # Add the specific prefix for mxbai-embed-large queries
    retrieval_query_text = query_text # Default
    if EMBEDDING_MODEL_NAME == 'mxbai-embed-large':
        retrieval_query_text = f"Represent this sentence for searching relevant passages: {query_text}"
        print(f"Using mxbai-specific query for retrieval: '{retrieval_query_text[:80]}...'")
    
    # This function aims to return scores between 0 and 1 (higher is better)
    results_with_scores = vector_store.similarity_search_with_relevance_scores(retrieval_query_text, k=top_n)
    
    relevant_chunks = []
    for doc, score in results_with_scores:
        relevant_chunks.append({
            'text': doc.page_content,
            'source': doc.metadata.get('source', 'Unknown'),
            'score': score # Relevance score (0-1, higher is better)
        })
    # Results from similarity_search_with_relevance_scores should already be sorted by relevance
    return relevant_chunks


def generate_response_with_llm(query, relevant_chunks):
    """
    Generates a response using the LLM, augmented with relevant chunks.
    """
    context_str = "\n\n".join([f"Source: {chunk['source'].split(os.sep)[-1]}\nContent: {chunk['text']}" for chunk in relevant_chunks]) # Show only filename

    prompt = f"""You are a helpful assistant. Answer the following question based ONLY on the provided context.
The context may include tables of contents, chapter titles, or narrative text. Extract the answer if it is present.
If the information to answer the question is not in the context, you MUST respond with 'I don't know based on the provided context'.

Context:
{context_str}

Question: {query}

Answer:"""

    print(f"\n--- Sending prompt to LLM ({LLM_MODEL_NAME}) ---")
    print(f"Augmented Prompt (first 500 chars):\n{prompt[:500]}...\n")

    try:
        response = ollama.chat(
            model=LLM_MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    except Exception as e:
        print(f"Error generating response from LLM: {e}")
        return "Sorry, I encountered an error while trying to generate a response."

# --- Main RAG Pipeline ---
def run_rag_pipeline():
    # 1. Initialize or Load Knowledge Base (Vector Store)
    print("--- Step 1: Initializing/Loading Knowledge Base ---")
    vector_store = load_or_create_knowledge_base()
    if not vector_store:
        print("Failed to initialize knowledge base. Exiting.")
        return

    # 2. Interact with the user
    print("\n--- Step 2: Ready to answer questions ---")
    print("Type 'quit' or 'exit' to stop.")
    while True:
        user_query = input("\nAsk your question: ")
        if user_query.lower() in ['quit', 'exit']:
            break
        if not user_query.strip():
            continue

        # a. Find relevant chunks from vector store
        relevant_chunks = find_relevant_chunks_from_vectorstore(vector_store, user_query, top_n=4)

        if not relevant_chunks:
            print("No relevant context found for your query.")
            final_answer = "I couldn't find any relevant information in my current knowledge base to answer your question."
        else:
            # ...
            print(f"Found {len(relevant_chunks)} relevant chunk(s):")
            for i, chunk_info in enumerate(relevant_chunks):
                source_filename = chunk_info['source'].split(os.sep)[-1]
                print(f"  {i+1}. Source: {source_filename}, Relevance Score: {chunk_info['score']:.4f}")
                print(f"     Content Snippet: {chunk_info['text'][:200]}...") # Print a longer snippet or full text for debugging
            # ...

            # b. Generate response
            print("Generating answer...")
            final_answer = generate_response_with_llm(user_query, relevant_chunks)

        print("\n--- Answer ---")
        print(final_answer)

if __name__ == '__main__':
    run_rag_pipeline()