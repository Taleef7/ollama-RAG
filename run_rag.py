# Force Python to use the newer SQLite version from pysqlite3-binary
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("✅ Successfully swapped sqlite3 with pysqlite3.")
except ImportError:
    print("⚠️ pysqlite3-binary not found, using system sqlite3.")
except KeyError:
    print("⚠️ Error during pysqlite3 to sqlite3 module replacement.")

# --- Core Python and Utility Imports ---
import os
import time
from typing import List, Tuple, Any

# --- Environment Variable Management ---
from dotenv import load_dotenv
load_dotenv()

# --- LangChain Core, Community, and Specific Package Imports ---
from langchain_core.embeddings import Embeddings as LangChainEmbeddingsBase
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# --- Sentence Transformers Import (for Embedding Model) ---
from sentence_transformers import SentenceTransformer

# --- Hugging Face Transformers Imports (for LLM) ---
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch

# --- Configuration (Shared between CLI and Streamlit app if imported) ---
DOCUMENT_DIR = "data"
EMBEDDING_CHOICE = "linq-embed-mistral"
LINQ_EMBED_MODEL_ID = "Linq-AI-Research/Linq-Embed-Mistral"
LINQ_EMBED_QUERY_TASK_DESCRIPTION = "Given a user question, retrieve relevant text passages from the documents that can help answer the question."
EMBEDDING_DEVICE = "cuda"

HF_LLM_MODEL_ID = "google/gemma-3-12b-it" # Verify exact ID
LLM_DEVICE = "cuda" # device_map="auto" will handle actual placement

CHROMA_PERSIST_DIR = "chroma_db_linq_mistral"
COLLECTION_NAME = "rag_linq_mistral_gemma3_12b_collection"
EMBEDDING_MODEL_NAME_FOR_PRINT = LINQ_EMBED_MODEL_ID

CHUNK_SIZE = 2048
CHUNK_OVERLAP = 200

# --- Module-level cache for models (for CLI use, Streamlit will use its own caching) ---
_embedding_model_instance = None
_llm_model = None
_llm_processor = None

# --- Custom SentenceTransformer Embeddings Class for LangChain ---
class HFLocalSentenceTransformerEmbeddings(LangChainEmbeddingsBase):
    def __init__(self, model_id: str, query_task_description: str, device: str = EMBEDDING_DEVICE):
        self.model_id = model_id
        self.device = device
        self.query_prompt_prefix = f"Instruct: {query_task_description}\nQuery: "
        print(f"Initializing SentenceTransformer embedding model: {self.model_id} onto device: {self.device}...")
        self.model = SentenceTransformer(self.model_id, device=self.device, trust_remote_code=True)
        print(f"✅ SentenceTransformer embedding model '{self.model_id}' loaded successfully.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        print(f"Embedding {len(texts)} document chunks with {self.model_id}...")
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=16)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        print(f"Embedding query with {self.model_id} using prefix '{self.query_prompt_prefix[:30]}...': '{text[:60]}...'")
        # The 'prompt' argument in model.encode for SentenceTransformer prepends the string to EACH item in the first arg list.
        # So, for a single query string, we pass it as a list of one item.
        embedding = self.model.encode([text], prompt=self.query_prompt_prefix, batch_size=1)[0]
        return embedding.tolist()

# --- Model Loading Functions (for Streamlit to cache their results) ---
def get_embedding_model_instance() -> HFLocalSentenceTransformerEmbeddings:
    global _embedding_model_instance
    if _embedding_model_instance is None:
        print("RUN_RAG: Loading embedding model instance for the first time in this process...")
        _embedding_model_instance = HFLocalSentenceTransformerEmbeddings(
            model_id=LINQ_EMBED_MODEL_ID,
            query_task_description=LINQ_EMBED_QUERY_TASK_DESCRIPTION,
            device=EMBEDDING_DEVICE
        )
    return _embedding_model_instance

def get_llm_and_processor() -> Tuple[Any, Any]: # Returns (model, processor)
    global _llm_model, _llm_processor
    if _llm_model is None or _llm_processor is None:
        print(f"RUN_RAG: Loading Hugging Face Model & Processor: {HF_LLM_MODEL_ID} (this may take a while)...")
        _llm_processor = AutoProcessor.from_pretrained(HF_LLM_MODEL_ID, trust_remote_code=True)
        _llm_model = Gemma3ForConditionalGeneration.from_pretrained(
            HF_LLM_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
	    attn_implementation="eager"
        )
        _llm_model.eval()
        if hasattr(_llm_processor, 'tokenizer') and _llm_processor.tokenizer.pad_token_id is None:
            if _llm_processor.tokenizer.eos_token_id is not None:
                _llm_processor.tokenizer.pad_token_id = _llm_processor.tokenizer.eos_token_id
        print(f"✅ RUN_RAG: Hugging Face Model & Processor '{HF_LLM_MODEL_ID}' loaded.")
    return _llm_model, _llm_processor

# --- Knowledge Base Function ---
def load_or_create_knowledge_base(embedding_function: HFLocalSentenceTransformerEmbeddings):
    # (Content of this function remains largely the same as your last working version)
    # Ensure it uses CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME_FOR_PRINT,
    # DOCUMENT_DIR, CHUNK_SIZE, CHUNK_OVERLAP from global config.
    if not embedding_function: print("🔴 Embedding function not provided."); return None
    # ... (rest of the function as in your last full script) ...
    if not os.path.exists(CHROMA_PERSIST_DIR): os.makedirs(CHROMA_PERSIST_DIR)
    is_existing_db = os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR)
    vector_store = None
    if is_existing_db:
        try:
            vector_store = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embedding_function, collection_name=COLLECTION_NAME)
            if vector_store._collection.count() == 0: is_existing_db = False; print("Warning: DB empty, rebuilding.")
            else: print(f"Vector store loaded. Collection '{COLLECTION_NAME}' has {vector_store._collection.count()} docs.")
        except Exception as e: is_existing_db = False; print(f"🔴 Error loading DB: {e}. Rebuilding.")
    if not is_existing_db:
        print(f"Creating new vector store: {EMBEDDING_MODEL_NAME_FOR_PRINT}, Chunks: {CHUNK_SIZE}/{CHUNK_OVERLAP}")
        loader = DirectoryLoader(DOCUMENT_DIR, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'}, show_progress=True, use_multithreading=False)
        docs = loader.load(); print(f"Loaded {len(docs)} document(s).")
        if not docs: return None
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len)
        doc_chunks = text_splitter.split_documents(docs); print(f"Split into {len(doc_chunks)} chunks.")
        if not doc_chunks: return None
        print(f"Creating vector store with {len(doc_chunks)} chunks (this will take time)...")
        vector_store = Chroma.from_documents(documents=doc_chunks, embedding=embedding_function, collection_name=COLLECTION_NAME, persist_directory=CHROMA_PERSIST_DIR, collection_metadata={"hnsw:space": "cosine"})
        print(f"Vector store created. Collection has {vector_store._collection.count()} documents.")
    return vector_store


# --- Retrieval Function ---
def find_relevant_chunks_from_vectorstore(vector_store, query_text: str, top_n: int = 15):
    if not vector_store: print("🔴 Vector store not initialized."); return []
    print(f"\nSearching for relevant chunks for query: '{query_text[:60]}...' (top_n={top_n})")
    return vector_store.similarity_search_with_relevance_scores(query_text, k=top_n)

# --- LLM Generation Function (now takes model and processor as args) ---
def generate_response_with_llm(question_for_llm: str, context_str_for_llm: str, current_llm_model, current_llm_processor):
    if not current_llm_model or not current_llm_processor: return "Error: LLM model/processor not provided."

    system_message_content = """You are a precision-focused Q&A engine. Your sole task is to accurately answer the user's question based *only and entirely* on the provided context.
Key Instructions:
1.  **Analyze Thoroughly:** Scrutinize all provided context snippets.
2.  **Formulating Your Answer:** Direct Extraction, Synthesis & Inference (strongly supported by text).
3.  **Strict Contextual Grounding:** Base answer *solely* on context. No external knowledge.
4.  **Handling Unanswerable Questions:** If info is definitively not in context, respond *only* with: "I don't know based on the provided context"."""
    user_message_content = f"""**Provided Context:**\n---\n{context_str_for_llm}\n---\n\n**Question:** {question_for_llm}"""
    messages = [{"role": "system", "content": [{"type": "text", "text": system_message_content}]},
                {"role": "user", "content": [{"type": "text", "text": user_message_content}]}]
    
    print(f"\n--- Preparing prompt for {HF_LLM_MODEL_ID} using apply_chat_template ---")
    try:
        inputs = current_llm_processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(current_llm_model.device)
    except Exception as e: print(f"🔴 Error applying chat template: {e}"); return "Error in prompt prep."

    input_len = inputs["input_ids"].shape[-1]
    print(f"--- Generating response with {HF_LLM_MODEL_ID} (input length: {input_len} tokens) ---")
    try:
        with torch.inference_mode():
            generation_outputs = current_llm_model.generate(**inputs, max_new_tokens=1024, do_sample=False, top_k=None, top_p=None) # Added top_k, top_p
        generated_ids = generation_outputs[0][input_len:]
        generated_text = current_llm_processor.decode(generated_ids, skip_special_tokens=True)
        return generated_text.strip()
    except Exception as e: print(f"🔴 Error generating response: {e}"); import traceback; traceback.print_exc(); return "Error during LLM generation."

# --- Main RAG Pipeline (for CLI execution) ---
def run_rag_pipeline_cli():
    print("--- Initializing Models for CLI ---")
    try:
        embedding_function = get_embedding_model_instance()
        if not embedding_function: raise ValueError("Embedding model could not be loaded.")
        
        llm, processor = get_llm_and_processor()
        if not llm or not processor: raise ValueError("LLM or Processor could not be loaded.")
    except Exception as e:
        print(f"🔴 Failed to initialize models for CLI: {e}")
        return

    print("\n--- Step 1: Initializing/Loading Knowledge Base (CLI) ---")
    vector_store = load_or_create_knowledge_base(embedding_function)
    if not vector_store: print("🔴 Failed to init knowledge base. Exiting."); return

    print("\n--- Step 2: Ready to answer questions (CLI) ---")
    print(f"--- Using LLM: {HF_LLM_MODEL_ID} ---")
    print("Type 'quit' or 'exit' to stop.")
    while True:
        try: user_query = input("\nAsk your question: ")
        except KeyboardInterrupt: print("\nExiting..."); break
        if user_query.lower() in ['quit', 'exit']: break
        if not user_query.strip(): continue

        retrieved_chunks_with_scores = find_relevant_chunks_from_vectorstore(vector_store, user_query, top_n=15)
        if not retrieved_chunks_with_scores:
            final_answer = "I couldn't find any relevant information for your question."
        else:
            print(f"Found {len(retrieved_chunks_with_scores)} relevant chunk(s):")
            context_for_llm_parts = []
            for i, (doc, score) in enumerate(retrieved_chunks_with_scores): # Iterate over (doc, score)
                source_filename = os.path.basename(doc.metadata.get('source', 'Unknown'))
                score_display = f"{score:.4f}" if isinstance(score, float) else "N/A"
                print(f"  {i+1}. Source: {source_filename}, Relevance Score: {score_display}, Snippet: {doc.page_content[:100].strip()}...")
                context_for_llm_parts.append(f"Source: {source_filename}\nContent: {doc.page_content}")
            context_str_for_llm = "\n\n---\n\n".join(context_for_llm_parts)
            
            if "python" in user_query.lower() and "philosophy" in user_query.lower(): # Temporary debug print
                print(f"\n--- Full Context Being Sent to LLM for Python query ---\n{context_str_for_llm}\n--- End of Full Context ---")

            print("Generating answer...")
            final_answer = generate_response_with_llm(user_query, context_str_for_llm, llm, processor) # Pass llm and processor
        print("\n--- Answer ---"); print(final_answer)

if __name__ == '__main__':
    print("Script `run_rag.py` starting via __main__ (CLI mode)...")
    try:
        run_rag_pipeline_cli() # Call the CLI specific pipeline
    except Exception as e_main:
        print(f"🔴 Main execution error in CLI mode: {e_main}")
        import traceback
        traceback.print_exc()
    finally:
        print("Script `run_rag.py` finished CLI execution.")
