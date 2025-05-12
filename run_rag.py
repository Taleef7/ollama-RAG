import ollama
import os
import numpy as np

# --- Configuration ---
DOCUMENT_DIR = "data"  # Directory where your text files are stored
EMBEDDING_MODEL = 'mxbai-embed-large' # The model you tested for embeddings
LLM_MODEL = 'gemma3:1b-it-qat'      # The LLM you tested for chat

# --- Global store for our knowledge base (chunks and their embeddings) ---
# In a real application, you'd use a vector database (ChromaDB, FAISS, etc.)
knowledge_base = [] # List of dictionaries: {'text': chunk_text, 'embedding': vector}

# --- Helper Function for Cosine Similarity ---
def cosine_similarity(v1, v2):
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0 # Avoid division by zero
    return dot_product / (norm_v1 * norm_v2)

# --- Phase 1: Building the Knowledge Base ---
def load_and_chunk_documents(directory):
    """
    Loads documents from the specified directory and chunks them.
    For simplicity, each file is treated as a single chunk.
    More sophisticated chunking can be added later.
    """
    chunks = []
    print(f"Loading documents from: {directory}")
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                    if text.strip(): # Ensure the chunk is not empty
                        chunks.append({'text': text, 'source': filename})
                        print(f"  Loaded and chunked: {filename}")
                    else:
                        print(f"  Skipped empty file: {filename}")
            except Exception as e:
                print(f"  Error loading file {filename}: {e}")
    return chunks

def generate_embeddings_for_chunks(chunks):
    """
    Generates embeddings for each chunk using the specified Ollama model
    and stores them in the global knowledge_base.
    """
    global knowledge_base
    knowledge_base = [] # Clear existing base if any
    print(f"\nGenerating embeddings using model: {EMBEDDING_MODEL}")
    for i, chunk_doc in enumerate(chunks):
        text = chunk_doc['text']
        try:
            response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
            knowledge_base.append({
                'text': text,
                'embedding': response['embedding'],
                'source': chunk_doc['source']
            })
            print(f"  Generated embedding for chunk {i+1} from {chunk_doc['source']} (vector length: {len(response['embedding'])})")
        except Exception as e:
            print(f"  Error generating embedding for chunk from {chunk_doc['source']}: {e}")
    print(f"Knowledge base built. Total chunks: {len(knowledge_base)}")

# --- Phase 2: Retrieval and Generation ---
def find_relevant_chunks(query_embedding, top_n=2):
    """
    Finds the most relevant chunks from the knowledge base using cosine similarity.
    """
    if not knowledge_base:
        print("Knowledge base is empty. Cannot find relevant chunks.")
        return []

    similarities = []
    for entry in knowledge_base:
        similarity = cosine_similarity(query_embedding, entry['embedding'])
        similarities.append({'text': entry['text'], 'source': entry['source'], 'score': similarity})

    # Sort by similarity score in descending order
    similarities.sort(key=lambda x: x['score'], reverse=True)

    return similarities[:top_n]

def generate_response(query, relevant_chunks):
    """
    Generates a response using the LLM, augmented with relevant chunks.
    """
    context_str = "\n\n".join([f"Source: {chunk['source']}\nContent: {chunk['text']}" for chunk in relevant_chunks])

    prompt = f"""Based on the following context, please answer the question.
If the context does not contain the answer, say 'I don't know based on the provided context'.

Context:
{context_str}

Question: {query}

Answer:"""

    print(f"\n--- Sending prompt to LLM ({LLM_MODEL}) ---")
    print(f"Augmented Prompt:\n{prompt[:500]}...\n") # Print beginning of prompt for brevity

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    except Exception as e:
        print(f"Error generating response from LLM: {e}")
        return "Sorry, I encountered an error while trying to generate a response."

# --- Main RAG Pipeline ---
def run_rag_pipeline():
    # 1. Build Knowledge Base
    print("--- Step 1: Building Knowledge Base ---")
    doc_chunks = load_and_chunk_documents(DOCUMENT_DIR)
    if not doc_chunks:
        print("No documents found or loaded. Exiting.")
        return
    generate_embeddings_for_chunks(doc_chunks)
    if not knowledge_base:
        print("Knowledge base construction failed. Exiting.")
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

        # a. Embed the query
        print(f"Embedding your query using {EMBEDDING_MODEL}...")
        try:
            query_embedding_response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=user_query)
            query_embedding = query_embedding_response['embedding']
            print("Query embedded successfully.")
        except Exception as e:
            print(f"Error embedding query: {e}")
            continue

        # b. Find relevant chunks
        print("Finding relevant chunks from knowledge base...")
        relevant_chunks = find_relevant_chunks(query_embedding, top_n=2)

        if not relevant_chunks:
            print("No relevant context found for your query.")
            final_answer = "I couldn't find any relevant information in my current knowledge base to answer your question."
        else:
            print(f"Found {len(relevant_chunks)} relevant chunk(s):")
            for i, chunk_info in enumerate(relevant_chunks):
                print(f"  {i+1}. Source: {chunk_info['source']}, Score: {chunk_info['score']:.4f}")
                # print(f"     Content: {chunk_info['text'][:100]}...") # Optionally print snippet of chunk

            # c. Generate response
            print("Generating answer...")
            final_answer = generate_response(user_query, relevant_chunks)

        print("\n--- Answer ---")
        print(final_answer)

if __name__ == '__main__':
    run_rag_pipeline()