# --- Core Python and Utility Imports ---
import os
import time
import numpy as np # Still useful for cosine_similarity if used elsewhere, though Chroma handles it
from typing import List, Dict, Any # For type hinting

# --- Environment Variable Management ---
from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file at the very beginning

# --- Google Gemini API Imports ---
import google.generativeai as genai

# --- LangChain Core, Community, and Specific Package Imports ---
from langchain_core.embeddings import Embeddings as LangChainEmbeddingsBase # Renamed to avoid conflict
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter # Using from langchain.text_splitter
from langchain_chroma import Chroma # Corrected import for Chroma
# Note: OllamaEmbeddings is no longer needed if fully switching to Gemini for embeddings
# from langchain_ollama import OllamaEmbeddings

# --- Ollama Client (for LLM interaction) ---
import ollama


# --- Configuration ---
DOCUMENT_DIR = "data"
# Use a new directory for ChromaDB with Gemini embeddings
CHROMA_PERSIST_DIR = "chroma_db_gemini"
COLLECTION_NAME = "rag_gemini_collection"

# LLM model remains from Ollama
LLM_MODEL_NAME = 'gemma3:4b-it-qat'
# Gemini Embedding Model ID for the API
GEMINI_EMBEDDING_MODEL_ID = "gemini-embedding-exp-03-07" # Or "models/gemini-embedding-exp-03-07"

# --- Gemini API Key Setup ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("🔴 FATAL: GEMINI_API_KEY not found in .env file. Please set it. Exiting.")
    exit() # Exit if key is not found
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("✅ Gemini API Key configured.")
    except Exception as e:
        print(f"🔴 FATAL: Error configuring Gemini API (check API key validity): {e}. Exiting.")
        exit()

# --- Custom Gemini Embeddings Class for LangChain ---
class GeminiLangChainEmbeddings(LangChainEmbeddingsBase):
    """
    Custom LangChain Embeddings wrapper for the Google Gemini API.
    Uses specific task_types for document and query embeddings.
    """
    def __init__(self, model_name: str = GEMINI_EMBEDDING_MODEL_ID):
        self.model_name = model_name
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not configured for GeminiLangChainEmbeddings.")

    # This is the method that actually calls the Gemini API for a batch
    def _call_gemini_embed_content(self, texts_batch: List[str], task_type: str) -> List[List[float]]:
        """Internal method to embed a single batch and handle API call errors."""
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=texts_batch,
                task_type=task_type
            )
            return result['embedding']
        except Exception as e:
            # This will catch the 429 error if it happens for this specific batch
            print(f"🔴 Error during genai.embed_content for a batch (task: {task_type}): {e}")
            raise e # Re-raise the exception to be caught by the calling method

    def _embed_with_batching_and_delay(self, texts: List[str], task_type: str,
                                       api_batch_size: int = 3,
                                       initial_delay_seconds: int = 25,
                                       subsequent_delay_seconds: int = 20) -> List[List[float]]:
        """Embeds texts in batches with delays to handle API rate limits."""
        all_embeddings: List[List[float]] = []

        if not texts: # Handle empty list of texts
            return []

        if initial_delay_seconds > 0:
            print(f"    Initial delay of {initial_delay_seconds}s before starting embedding process...")
            time.sleep(initial_delay_seconds)

        for i in range(0, len(texts), api_batch_size):
            batch_texts = texts[i:i + api_batch_size]
            num_batches = (len(texts) + api_batch_size - 1) // api_batch_size
            current_batch_num = i // api_batch_size + 1

            print(f"    Embedding batch {current_batch_num}/{num_batches} (size: {len(batch_texts)}) with Gemini ({task_type})...")

            try:
                # CORRECTED CALL: Use the new _call_gemini_embed_content method
                batch_embeddings = self._call_gemini_embed_content(batch_texts, task_type)
                all_embeddings.extend(batch_embeddings)

                if current_batch_num < num_batches:
                    print(f"    ...batch successful. Waiting {subsequent_delay_seconds}s before next batch...")
                    time.sleep(subsequent_delay_seconds)
                else:
                    print(f"    ...last batch successful.")
            except Exception as e:
                # Error already printed in _call_gemini_embed_content. Re-raise to stop processing.
                # The calling methods (embed_documents, embed_query) should handle this.
                print(f"    Problematic batch (first text snippet): {batch_texts[0][:100]}...")
                raise e # Propagate the error
        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        print(f"Embedding {len(texts)} document chunks with Gemini (task_type=RETRIEVAL_DOCUMENT)...")
        try:
            return self._embed_with_batching_and_delay(
                texts,
                task_type="RETRIEVAL_DOCUMENT",
                api_batch_size=3, # Start with a very small batch
                initial_delay_seconds=25,
                subsequent_delay_seconds=20
            )
        except Exception as e:
            print(f"🔴 Halting document embedding due to API error during batch processing. Error: {e}")
            raise e # Re-raise for Chroma.from_documents to catch

    def embed_query(self, text: str) -> List[float]:
        print(f"Embedding query with Gemini (task_type=RETRIEVAL_QUERY): '{text[:60]}...'")
        try:
            list_of_one_embedding = self._embed_with_batching_and_delay(
                [text],
                task_type="RETRIEVAL_QUERY",
                api_batch_size=3,
                initial_delay_seconds=25,
                subsequent_delay_seconds=20
            )
            return list_of_one_embedding[0] if list_of_one_embedding and list_of_one_embedding[0] else []
        except Exception as e:
            print(f"🔴 Halting query embedding due to API error. Error: {e}")
            return []

# --- Initialize Your Chosen Embedding Function ---
# Now we use the Gemini embeddings
if GEMINI_API_KEY:
    active_embedding_function = GeminiLangChainEmbeddings()
    print(f"✅ Using Gemini Embeddings: {GEMINI_EMBEDDING_MODEL_ID}")
else:
    # Fallback or error if Gemini key isn't available.
    # For now, we'll exit if key is missing, as handled above.
    # If you wanted a fallback to Ollama embeddings:
    # print("⚠️ Gemini API Key not found. Falling back to Ollama embeddings (mxbai-embed-large).")
    # active_embedding_function = OllamaEmbeddings(model='mxbai-embed-large')
    # EMBEDDING_MODEL_NAME_FOR_PRINT = 'mxbai-embed-large (Ollama fallback)'
    print("🔴 Script will exit as Gemini API key is required and not found.")
    exit()

EMBEDDING_MODEL_NAME_FOR_PRINT = GEMINI_EMBEDDING_MODEL_ID # For print statements

# --- Phase 1: Building or Loading the Knowledge Base ---
def load_or_create_knowledge_base():
    if not active_embedding_function:
        print("🔴 Embedding function not initialized. Cannot build knowledge base.")
        return None

    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        print(f"Loading existing vector store from: {CHROMA_PERSIST_DIR}")
        vector_store = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=active_embedding_function, # USE THE CHOSEN EMBEDDING FUNCTION
            collection_name=COLLECTION_NAME
        )
        print(f"Vector store loaded. Collection '{COLLECTION_NAME}' has {vector_store._collection.count()} documents.")
    else:
        print(f"Creating new vector store with {EMBEDDING_MODEL_NAME_FOR_PRINT} and COSINE similarity...")
        print(f"Loading documents from: {DOCUMENT_DIR}")
        loader_kwargs = {'encoding': 'utf-8'}
        loader = DirectoryLoader(
            DOCUMENT_DIR, glob="**/*.txt", loader_cls=TextLoader,
            loader_kwargs=loader_kwargs, show_progress=True, use_multithreading=True
        )
        print("Attempting to load documents with UTF-8 encoding...")
        try:
            documents = loader.load()
        except Exception as e:
            print(f"🔴 An error occurred during document loading: {e}")
            return None

        if not documents:
            print("No documents found. Cannot build knowledge base.")
            return None
        print(f"Loaded {len(documents)} document(s).")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=150, length_function=len
        )
        doc_chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(doc_chunks)} chunks.")

        if not doc_chunks:
            print("No chunks created. Cannot build knowledge base.")
            return None

        print(f"Creating vector store. This will use {EMBEDDING_MODEL_NAME_FOR_PRINT} for embeddings...")
        vector_store = Chroma.from_documents(
            documents=doc_chunks,
            embedding=active_embedding_function, # USE THE CHOSEN EMBEDDING FUNCTION
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"Vector store created with COSINE similarity and persisted at: {CHROMA_PERSIST_DIR}")
        print(f"Collection '{COLLECTION_NAME}' has {vector_store._collection.count()} documents.")
    return vector_store

# --- Phase 2: Retrieval and Generation ---
def find_relevant_chunks_from_vectorstore(vector_store, query_text, top_n=4): # Using top_n=4 from last test
    if not vector_store:
        print("🔴 Vector store is not initialized.")
        return []

    # The custom GeminiLangChainEmbeddings wrapper handles task types,
    # so no manual query prefixing is needed here.
    print(f"\nSearching for relevant chunks for query: '{query_text[:60]}...'")
    results_with_scores = vector_store.similarity_search_with_relevance_scores(query_text, k=top_n)

    relevant_chunks = []
    for doc, score in results_with_scores:
        relevant_chunks.append({
            'text': doc.page_content,
            'source': doc.metadata.get('source', 'Unknown'),
            'score': score
        })
    return relevant_chunks

def generate_response_with_llm(query, relevant_chunks):
    context_str = "\n\n".join([f"Source: {chunk['source'].split(os.sep)[-1]}\nContent: {chunk['text']}" for chunk in relevant_chunks])

    prompt = f"""You are a helpful assistant. Answer the following question based ONLY on the provided context.
The context may include tables of contents, chapter titles, or narrative text. Extract the answer if it is present.
If the information to answer the question is not in the context, you MUST respond with 'I don't know based on the provided context'.

Context:
{context_str}

Question: {query}

Answer:"""

    print(f"\n--- Sending prompt to LLM ({LLM_MODEL_NAME}) ---")
    # Ensure Augmented Prompt is not excessively long for printing
    max_prompt_print_len = 700
    printable_prompt = prompt
    if len(prompt) > max_prompt_print_len:
        printable_prompt = prompt[:max_prompt_print_len] + "\n[...rest of prompt truncated for display...]"
    print(f"Augmented Prompt:\n{printable_prompt}\n")


    try:
        response = ollama.chat(
            model=LLM_MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    except Exception as e:
        print(f"🔴 Error generating response from LLM: {e}")
        return "Sorry, I encountered an error while trying to generate a response from the LLM."

# --- Main RAG Pipeline ---
def run_rag_pipeline():
    print("--- Step 1: Initializing/Loading Knowledge Base ---")
    vector_store = load_or_create_knowledge_base()
    if not vector_store:
        print("🔴 Failed to initialize knowledge base. Exiting.")
        return

    print("\n--- Step 2: Ready to answer questions ---")
    print("Type 'quit' or 'exit' to stop.")
    while True:
        try:
            user_query = input("\nAsk your question: ")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

        if user_query.lower() in ['quit', 'exit']:
            break
        if not user_query.strip():
            continue

        relevant_chunks = find_relevant_chunks_from_vectorstore(vector_store, user_query, top_n=4)

        if not relevant_chunks:
            # Check if any chunk had empty embeddings (could happen if Gemini API failed for some)
            print("No relevant context found for your query, or embeddings might have failed for some chunks.")
            final_answer = "I couldn't find any relevant information in my current knowledge base to answer your question, or there was an issue processing some data."
        else:
            print(f"Found {len(relevant_chunks)} relevant chunk(s):")
            for i, chunk_info in enumerate(relevant_chunks):
                source_filename = chunk_info['source'].split(os.sep)[-1]
                # Ensure score is a float before formatting, handle None case
                score_display = f"{chunk_info['score']:.4f}" if isinstance(chunk_info['score'], float) else "N/A"
                print(f"  {i+1}. Source: {source_filename}, Relevance Score: {score_display} (higher is better for relevance_scores)")
                # Print the full chunk text for debugging if desired (as you had before)
                print(f"     --- Start of Chunk {i+1} Content ---")
                print(chunk_info['text'])
                print(f"     --- End of Chunk {i+1} Content ---")
                print(f"     Content Snippet: {chunk_info['text'][:200]}...")


            print("Generating answer...")
            final_answer = generate_response_with_llm(user_query, relevant_chunks)

        print("\n--- Answer ---")
        print(final_answer)

if __name__ == '__main__':
    if GEMINI_API_KEY: # Only run if API key is present
        run_rag_pipeline()
    else:
        print("🔴 RAG pipeline cannot run because GEMINI_API_KEY is not set.")