# --- Core Python and Utility Imports ---
import os
import time
import numpy as np
from typing import List, Dict, Any

# --- Environment Variable Management ---
from dotenv import load_dotenv
load_dotenv()

# --- Google Gemini API Imports (Conditional) ---
import google.generativeai as genai

# --- LangChain Core, Community, and Specific Package Imports ---
from langchain_core.embeddings import Embeddings as LangChainEmbeddingsBase
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# --- Sentence Transformers Import (for Stella) ---
# Make sure to install it: pip install sentence-transformers
from sentence_transformers import SentenceTransformer

# --- Ollama Client (for LLM interaction) ---
import ollama

# --- Configuration ---
DOCUMENT_DIR = "data"

# NEW: Choose your embedding model
# Options: "gemini" or "stella"
EMBEDDING_CHOICE = "stella" # <--- CHANGE HERE TO SWITCH

# LLM model remains from Ollama
LLM_MODEL_NAME = 'gemma3:4b-it-qat'

# Gemini Embedding Model ID (if EMBEDDING_CHOICE is "gemini")
GEMINI_EMBEDDING_MODEL_ID = "models/embedding-001" # Using the newer GA model ID "models/embedding-001" for general use, was "gemini-embedding-exp-03-07"

# Stella Embedding Model ID (if EMBEDDING_CHOICE is "stella")
STELLA_MODEL_ID = "NovaSearch/stella_en_400M_v5"
STELLA_QUERY_PROMPT_NAME = "s2p_query" # As per Stella's documentation for retrieval
STELLA_DEVICE = "cpu" # "cuda" for GPU, "cpu" for CPU

# Dynamic Chroma DB directory and collection name based on embedding choice
if EMBEDDING_CHOICE == "gemini":
    CHROMA_PERSIST_DIR = "chroma_db_gemini_v2" # Changed to v2 for clarity with new model
    COLLECTION_NAME = "rag_gemini_collection_v2"
    EMBEDDING_MODEL_NAME_FOR_PRINT = GEMINI_EMBEDDING_MODEL_ID
elif EMBEDDING_CHOICE == "stella":
    CHROMA_PERSIST_DIR = "chroma_db_stella"
    COLLECTION_NAME = "rag_stella_collection"
    EMBEDDING_MODEL_NAME_FOR_PRINT = STELLA_MODEL_ID
else:
    print(f"🔴 FATAL: Unknown EMBEDDING_CHOICE: {EMBEDDING_CHOICE}. Please choose 'gemini' or 'stella'. Exiting.")
    exit()

# --- Gemini API Key Setup (Conditional) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if EMBEDDING_CHOICE == "gemini":
    if not GEMINI_API_KEY:
        print("🔴 FATAL: GEMINI_API_KEY not found in .env file, but EMBEDDING_CHOICE is 'gemini'. Please set it. Exiting.")
        exit()
    else:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            print("✅ Gemini API Key configured.")
        except Exception as e:
            print(f"🔴 FATAL: Error configuring Gemini API (check API key validity): {e}. Exiting.")
            exit()

# --- Custom Gemini Embeddings Class for LangChain (from your script) ---
class GeminiLangChainEmbeddings(LangChainEmbeddingsBase):
    def __init__(self, model_name: str = GEMINI_EMBEDDING_MODEL_ID):
        self.model_name = model_name
        if not GEMINI_API_KEY: # Should be checked before instantiation, but good practice
            raise ValueError("GEMINI_API_KEY not configured for GeminiLangChainEmbeddings.")

    def _call_gemini_embed_content(self, texts_batch: List[str], task_type: str) -> List[List[float]]:
        try:
            # Note: "models/embedding-001" might not use title or output_dimensionality like "gemini-embedding-exp-03-07" did.
            # Adjust if using experimental models. For "models/embedding-001", task_type is main.
            # The model "models/embedding-001" uses task_type: "RETRIEVAL_DOCUMENT" or "RETRIEVAL_QUERY"
            # "SEMANTIC_SIMILARITY", "CLASSIFICATION", "CLUSTERING"
            params = {
                "model": self.model_name,
                "content": texts_batch,
                "task_type": task_type
            }
            if self.model_name == "gemini-embedding-exp-03-07": # Example for older experimental model
                 params["title"] = "Custom RAG Financial Query" # Optional title

            result = genai.embed_content(**params)
            return result['embedding']
        except Exception as e:
            print(f"🔴 Error during genai.embed_content for a batch (task: {task_type}): {e}")
            raise e

    def _embed_with_batching_and_delay(self, texts: List[str], task_type: str,
                                       api_batch_size: int = 90, # Gemini API allows up to 100, be safe
                                       initial_delay_seconds: int = 1,
                                       subsequent_delay_seconds: int = 1) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        if not texts: return []

        # For Gemini, the actual limit is RPM (Requests Per Minute), not just batch size.
        # The new models are more generous (e.g., embedding-001 has 1500 RPM).
        # Batching is good for efficiency, delay for strict RPM if hit.
        # Let's simplify: a small delay per batch if needed, but focus on API's batch limits.

        for i in range(0, len(texts), api_batch_size):
            batch_texts = texts[i:i + api_batch_size]
            num_batches = (len(texts) + api_batch_size - 1) // api_batch_size
            current_batch_num = i // api_batch_size + 1

            print(f"    Embedding batch {current_batch_num}/{num_batches} (size: {len(batch_texts)}) with Gemini ({task_type})...")
            if i > 0 and subsequent_delay_seconds > 0: # Delay for subsequent batches if specified
                time.sleep(subsequent_delay_seconds)
            elif i == 0 and initial_delay_seconds > 0: # Initial delay if specified
                 time.sleep(initial_delay_seconds)

            try:
                batch_embeddings = self._call_gemini_embed_content(batch_texts, task_type)
                all_embeddings.extend(batch_embeddings)
                print(f"    ...batch successful.")
            except Exception as e:
                print(f"    Problematic batch (first text snippet): {batch_texts[0][:100]}...")
                raise e
        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        print(f"Embedding {len(texts)} document chunks with Gemini (task_type=RETRIEVAL_DOCUMENT)...")
        try:
            # Using "RETRIEVAL_DOCUMENT" for the standard "models/embedding-001"
            return self._embed_with_batching_and_delay(
                texts,
                task_type="RETRIEVAL_DOCUMENT",
                api_batch_size=90, # Gemini API can handle up to 100 texts per call for embedding-001
                initial_delay_seconds=0, # Often not needed if RPM is high
                subsequent_delay_seconds=0 # Often not needed
            )
        except Exception as e:
            print(f"🔴 Halting document embedding due to API error. Error: {e}")
            raise e

    def embed_query(self, text: str) -> List[float]:
        print(f"Embedding query with Gemini (task_type=RETRIEVAL_QUERY): '{text[:60]}...'")
        try:
            # Using "RETRIEVAL_QUERY" for "models/embedding-001"
            list_of_one_embedding = self._embed_with_batching_and_delay(
                [text],
                task_type="RETRIEVAL_QUERY",
                api_batch_size=1, # Single query
                initial_delay_seconds=0,
                subsequent_delay_seconds=0
            )
            return list_of_one_embedding[0] if list_of_one_embedding and list_of_one_embedding[0] else []
        except Exception as e:
            print(f"🔴 Halting query embedding due to API error. Error: {e}")
            return []

# --- NEW: Custom Stella Embeddings Class for LangChain ---
class StellaLangChainEmbeddings(LangChainEmbeddingsBase):
    """
    Custom LangChain Embeddings wrapper for the NovaSearch/stella_en_400M_v5 model.
    Handles specific prompt_name for queries.
    """
    def __init__(self,
                 model_id: str = STELLA_MODEL_ID,
                 query_prompt_name: str = STELLA_QUERY_PROMPT_NAME,
                 device: str = STELLA_DEVICE):
        self.model_id = model_id
        self.query_prompt_name = query_prompt_name
        self.device = device
        try:
            print(f"Loading Stella model: {self.model_id} onto device: {self.device}...")

            # --- MODIFICATION START ---
            model_init_kwargs = {"trust_remote_code": True} # device is passed separately in SentenceTransformer >= 2.3.0
                                                         # For older versions it might be part of model_kwargs

            # According to Stella model card, for CPU usage if issues with xformers:
            if self.device == "cpu":
                model_init_kwargs["config_kwargs"] = {
                    "use_memory_efficient_attention": False,
                    "unpad_inputs": False
                }

            self.model = SentenceTransformer(
                self.model_id,
                device=self.device, # Pass device directly
                **model_init_kwargs # Pass other kwargs like trust_remote_code and config_kwargs
            )
            # --- MODIFICATION END ---

            print(f"✅ Stella model '{self.model_id}' loaded successfully on {self.device}.")
        except Exception as e:
            print(f"🔴 FATAL: Could not load Stella model '{self.model_id}'. Error: {e}")
            print("Ensure 'sentence-transformers' and 'torch' are installed, and the model ID is correct.")
            if "CUDA" in str(e).upper() and "xformers" not in str(e).lower(): # Check if CUDA error is not the xformers one
                print("    CUDA error detected. If you don't have a GPU or compatible CUDA, set STELLA_DEVICE='cpu'.")
            if "xformers" in str(e).lower() and self.device == "cpu":
                 print("    The model tried to use a feature requiring 'xformers'. The script attempted to disable it for CPU.")
                 print("    Ensure your sentence-transformers library is up to date: pip install -U sentence-transformers")
            elif "xformers" in str(e).lower() and self.device == "cuda":
                 print("    The model requires 'xformers' for GPU. Install it with: pip install xformers")

            raise e # Re-raise to stop execution if model loading fails

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        print(f"Embedding {len(texts)} document chunks with Stella model...")
        # For Stella, docs do not need any prompts according to their usage.
        # The SentenceTransformer's encode method handles batching internally and efficiently.
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32) # Default batch_size is 32
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        print(f"Embedding query with Stella model (prompt_name='{self.query_prompt_name}'): '{text[:60]}...'")
        # Queries use a specific prompt_name
        embedding = self.model.encode(text, prompt_name=self.query_prompt_name)
        return embedding.tolist()

# --- Initialize Your Chosen Embedding Function ---
active_embedding_function = None

if EMBEDDING_CHOICE == "gemini":
    if GEMINI_API_KEY:
        active_embedding_function = GeminiLangChainEmbeddings(model_name=GEMINI_EMBEDDING_MODEL_ID)
        print(f"✅ Using Gemini Embeddings: {GEMINI_EMBEDDING_MODEL_ID}")
    else:
        print("🔴 EMBEDDING_CHOICE is 'gemini' but GEMINI_API_KEY is not set. Exiting.")
        exit()
elif EMBEDDING_CHOICE == "stella":
    try:
        active_embedding_function = StellaLangChainEmbeddings(
            model_id=STELLA_MODEL_ID,
            query_prompt_name=STELLA_QUERY_PROMPT_NAME,
            device=STELLA_DEVICE
        )
        print(f"✅ Using Stella Embeddings: {STELLA_MODEL_ID} on {STELLA_DEVICE}")
    except Exception as e: # Catching potential model loading errors from constructor
        print(f"🔴 Failed to initialize Stella embeddings. Error: {e}. Exiting.")
        exit()
else: # Should have been caught earlier, but as a safeguard
    print(f"🔴 Invalid EMBEDDING_CHOICE: {EMBEDDING_CHOICE}. Exiting.")
    exit()


# --- Phase 1: Building or Loading the Knowledge Base ---
def load_or_create_knowledge_base():
    if not active_embedding_function:
        print("🔴 Embedding function not initialized. Cannot build knowledge base.")
        return None

    # Ensure the Chroma persist directory exists, or Chroma will create it
    if not os.path.exists(CHROMA_PERSIST_DIR):
        os.makedirs(CHROMA_PERSIST_DIR)
        print(f"Created Chroma persist directory: {CHROMA_PERSIST_DIR}")

    # Check if the directory is not empty AND contains Chroma files (basic check)
    # A more robust check would be to see if Chroma can actually load from it without error.
    is_existing_db = False
    if os.path.exists(CHROMA_PERSIST_DIR) and len(os.listdir(CHROMA_PERSIST_DIR)) > 0:
         # A simple check for some common Chroma files to guess if it's a valid DB
        expected_files = ["chroma.sqlite3"] # or files within a collection directory
        if any(f in os.listdir(CHROMA_PERSIST_DIR) for f in expected_files):
            is_existing_db = True
            print(f"Potential existing vector store found in: {CHROMA_PERSIST_DIR}")
        else:
            print(f"Directory {CHROMA_PERSIST_DIR} exists but doesn't look like a Chroma DB. Will attempt to create new.")


    if is_existing_db:
        print(f"Attempting to load existing vector store from: {CHROMA_PERSIST_DIR} with collection: {COLLECTION_NAME}")
        try:
            vector_store = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=active_embedding_function,
                collection_name=COLLECTION_NAME
            )
            # Verify collection has items
            if vector_store._collection.count() == 0:
                print(f"Warning: Loaded vector store, but collection '{COLLECTION_NAME}' is empty. Will try to rebuild.")
                # Treat as non-existing to force rebuild
                is_existing_db = False # Override to force creation
            else:
                print(f"Vector store loaded. Collection '{COLLECTION_NAME}' has {vector_store._collection.count()} documents.")
        except Exception as e:
            print(f"🔴 Error loading existing vector store: {e}. Will attempt to create a new one.")
            is_existing_db = False # Force creation

    if not is_existing_db: # Create new if not existing or loading failed or collection was empty
        print(f"Creating new vector store with {EMBEDDING_MODEL_NAME_FOR_PRINT} and COSINE similarity...")
        print(f"Loading documents from: {DOCUMENT_DIR}")
        loader_kwargs = {'encoding': 'utf-8'}
        loader = DirectoryLoader(
            DOCUMENT_DIR, glob="**/*.txt", loader_cls=TextLoader,
            loader_kwargs=loader_kwargs, show_progress=True, use_multithreading=False # DEBUG: turned off multithreading
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
            chunk_size=512, chunk_overlap=100, length_function=len
        )
        doc_chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(doc_chunks)} chunks.")

        if not doc_chunks:
            print("No chunks created. Cannot build knowledge base.")
            return None

        print(f"Creating vector store. This will use {EMBEDDING_MODEL_NAME_FOR_PRINT} for embeddings...")
        try:
            vector_store = Chroma.from_documents(
                documents=doc_chunks,
                embedding=active_embedding_function,
                collection_name=COLLECTION_NAME,
                persist_directory=CHROMA_PERSIST_DIR,
                collection_metadata={"hnsw:space": "cosine"} # Stella usually uses cosine
            )
            print(f"Vector store created with COSINE similarity and persisted at: {CHROMA_PERSIST_DIR}")
            print(f"Collection '{COLLECTION_NAME}' has {vector_store._collection.count()} documents.")
        except Exception as e:
            print(f"🔴🔴🔴 FATAL ERROR during Chroma.from_documents: {e}")
            print("This can happen if embedding generation fails for all documents (e.g., API key issue, model loading issue).")
            return None
    return vector_store

# --- Phase 2: Retrieval and Generation (largely unchanged) ---
def find_relevant_chunks_from_vectorstore(vector_store, query_text, top_n=8):
    if not vector_store:
        print("🔴 Vector store is not initialized.")
        return []

    print(f"\nSearching for relevant chunks for query: '{query_text[:50]}...'")
    # The custom embedding classes handle their specific needs (like prompts or task types)
    results_with_scores = vector_store.similarity_search_with_relevance_scores(query_text, k=top_n)

    relevant_chunks = []
    for doc, score in results_with_scores:
        relevant_chunks.append({
            'text': doc.page_content,
            'source': doc.metadata.get('source', 'Unknown'),
            'score': score # For similarity_search_with_relevance_scores, higher is better (more relevant)
        })
    return relevant_chunks

def generate_response_with_llm(query, relevant_chunks):
    context_str = "\n\n".join([f"Source: {chunk['source'].split(os.sep)[-1]}\nContent: {chunk['text']}" for chunk in relevant_chunks])

    prompt = f"""You are a precision-focused Q&A engine. Your sole task is to accurately answer the user's question based *only and entirely* on the provided context.

**Key Instructions:**

1.  **Analyze Thoroughly:**
    * Carefully read and understand the user's **question**: "{query}"
    * Scrutinize all provided **context snippets** below. The context may include tables of contents, chapter titles, narrative text, or other document excerpts.

2.  **Formulating Your Answer:**
    * **Direct Extraction:** If the answer is explicitly stated in the context, extract it precisely.
    * **Synthesis & Inference:** If the answer is not explicitly stated but can be logically derived or inferred by combining information from one or more context snippets, or by reasoning *directly* from the provided text, then formulate the answer.
        * If making an inference, ensure it is strongly supported by the text. You may optionally and briefly state your reasoning if the inference is complex (e.g., "Based on snippet A stating X and snippet B stating Y, it can be inferred that Z.").
    * **Completeness:** If multiple pieces of information from the context contribute to a comprehensive answer, synthesize them.

3.  **Strict Contextual Grounding (Crucial):**
    * Your answer MUST be based *solely* on the information found within the provided context snippets.
    * Do NOT use any external knowledge, prior training, or assumptions outside of what is explicitly given in the context.
    * If the context is ambiguous or seems to contain contradictory information relevant to the question, answer based on the most direct interpretation. If critical, you can briefly note the ambiguity as presented *in the context*.

4.  **Handling Unanswerable Questions:**
    * If, after careful and thorough analysis, you determine that the information required to answer the question is *definitively not present* or cannot be logically inferred from the provided context, you MUST respond with the exact phrase: "I don't know based on the provided context".
    * Do not add any apologies, explanations, or additional phrases to this specific response.

Context:
{context_str}

Question: {query}

Answer:"""

    print(f"\n--- Sending prompt to LLM ({LLM_MODEL_NAME}) ---")
    max_prompt_print_len = 700
    printable_prompt = prompt
    if len(prompt) > max_prompt_print_len:
        printable_prompt = prompt[:max_prompt_print_len] + "\n[...rest of prompt truncated for display...]"
    # print(f"Augmented Prompt:\n{printable_prompt}\n")

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
    print(f"--- Using {EMBEDDING_CHOICE.upper()} embeddings ---")
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

        relevant_chunks = find_relevant_chunks_from_vectorstore(vector_store, user_query, top_n=8)

        if not relevant_chunks:
            print("No relevant context found for your query.")
            final_answer = "I couldn't find any relevant information in my current knowledge base to answer your question."
        else:
            print(f"Found {len(relevant_chunks)} relevant chunk(s):")
            for i, chunk_info in enumerate(relevant_chunks):
                source_filename = os.path.basename(chunk_info['source']) # Cleaner way to get filename
                score_display = f"{chunk_info['score']:.4f}" if isinstance(chunk_info['score'], float) else "N/A"
                print(f"  {i+1}. Source: {source_filename}, Relevance Score: {score_display}")
                print(f"     Content Snippet: {chunk_info['text'][:200].strip()}...")
                # Uncomment below if you want to see full chunk text during debug
                # print(f"      --- Start of Chunk {i+1} Content ---")
                # print(chunk_info['text'])
                # print(f"      --- End of Chunk {i+1} Content ---")


            print("Generating answer...")
            final_answer = generate_response_with_llm(user_query, relevant_chunks)

        print("\n--- Answer ---")
        print(final_answer)

if __name__ == '__main__':
    # Basic check before running the full pipeline
    if EMBEDDING_CHOICE == "gemini" and not GEMINI_API_KEY:
        print("🔴 RAG pipeline cannot run because EMBEDDING_CHOICE is 'gemini' and GEMINI_API_KEY is not set.")
    elif active_embedding_function is None: # This implies Stella failed to load if chosen
        print("🔴 RAG pipeline cannot run because the embedding function could not be initialized.")
    else:
        run_rag_pipeline()