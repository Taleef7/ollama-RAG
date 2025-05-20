# Force Python to use the newer SQLite version from pysqlite3-binary
# This must be at the VERY TOP of your script, before any other imports
# that might indirectly import sqlite3 (like chromadb)
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("✅ Successfully swapped sqlite3 with pysqlite3.")
except ImportError:
    print("⚠️ pysqlite3-binary not found, using system sqlite3. This might lead to errors with ChromaDB if system sqlite3 is too old.")
except KeyError:
    print("⚠️ Error during pysqlite3 to sqlite3 module replacement. pysqlite3 might have already been replaced or an unusual state occurred.")

# --- Core Python and Utility Imports ---
import os
import time
from typing import List, Dict, Any # For type hinting

# --- Environment Variable Management ---
from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file

# --- LangChain Core, Community, and Specific Package Imports ---
from langchain_core.embeddings import Embeddings as LangChainEmbeddingsBase
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# --- Sentence Transformers Import (for Stella) ---
from sentence_transformers import SentenceTransformer

# --- Hugging Face Transformers Imports (for LLM) ---
from transformers import AutoProcessor, Gemma3ForConditionalGeneration # Using Gemma3 specific class
import torch

# --- Configuration ---
DOCUMENT_DIR = "data"

# --- Embedding Model Configuration (Stella) ---
EMBEDDING_CHOICE = "stella" # Keeping Stella as the embedding model
STELLA_MODEL_ID = "NovaSearch/stella_en_400M_v5"
STELLA_QUERY_PROMPT_NAME = "s2p_query"
STELLA_DEVICE = "cuda" # For Gilbreth GPU usage

# --- LLM Configuration (Hugging Face Gemma 3 12B-IT) ---
# IMPORTANT: Verify this model ID on Hugging Face. Instruction-tuned models are preferred.
HF_LLM_MODEL_ID = "google/gemma-3-12b-it"
LLM_DEVICE = "cuda" # For GPU acceleration on Gilbreth (device_map="auto" will handle placement)

# Chroma DB directory and collection name (assuming Stella embeddings for now)
CHROMA_PERSIST_DIR = "chroma_db_stella_hf_llm" # New DB dir for this version
COLLECTION_NAME = "rag_stella_gemma3_12b_collection"
EMBEDDING_MODEL_NAME_FOR_PRINT = STELLA_MODEL_ID

# --- Global placeholders for loaded models (to load only once) ---
embedding_model_instance = None # For Stella
llm_model = None                # For HF LLM model
llm_processor = None            # For HF LLM processor (replaces tokenizer for Gemma 3 multimodal)

# --- Custom Stella Embeddings Class for LangChain ---
class StellaLangChainEmbeddings(LangChainEmbeddingsBase):
    def __init__(self,
                 model_id: str = STELLA_MODEL_ID,
                 query_prompt_name: str = STELLA_QUERY_PROMPT_NAME,
                 device: str = STELLA_DEVICE):
        self.model_id = model_id
        self.query_prompt_name = query_prompt_name
        self.device = device
        try:
            print(f"Loading Stella embedding model: {self.model_id} onto device: {self.device}...")
            model_init_kwargs = {"trust_remote_code": True}
            if self.device == "cpu": # Robustness for CPU execution if needed
                model_init_kwargs["config_kwargs"] = {
                    "use_memory_efficient_attention": False,
                    "unpad_inputs": False
                }
            self.model = SentenceTransformer(
                self.model_id,
                device=self.device,
                **model_init_kwargs
            )
            print(f"✅ Stella embedding model '{self.model_id}' loaded successfully on {self.device}.")
        except Exception as e:
            print(f"🔴 FATAL: Could not load Stella embedding model '{self.model_id}'. Error: {e}")
            raise e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        print(f"Embedding {len(texts)} document chunks with Stella model...")
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        print(f"Embedding query with Stella model (prompt_name='{self.query_prompt_name}'): '{text[:60]}...'")
        embedding = self.model.encode(text, prompt_name=self.query_prompt_name)
        return embedding.tolist()

# --- Function to load Hugging Face LLM and Processor (once) ---
def load_hf_llm_and_processor(): # Renamed to processor
    global llm_model, llm_processor # Using llm_processor for clarity
    if llm_model is None or llm_processor is None:
        print(f"Loading Hugging Face Model & Processor: {HF_LLM_MODEL_ID} (this may take a while for large models)...")
        try:
            llm_processor = AutoProcessor.from_pretrained(HF_LLM_MODEL_ID, trust_remote_code=True)
            llm_model = Gemma3ForConditionalGeneration.from_pretrained( # Using Gemma3 specific class
                HF_LLM_MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="auto", # Automatically maps model to available GPU(s)
                trust_remote_code=True
                # Add attn_implementation="flash_attention_2" if supported and want to try for speed/memory
                # but ensure flash-attn is correctly installed in your venv: pip install flash-attn
            )
            llm_model.eval() # Set model to evaluation mode

            print(f"✅ Hugging Face Model & Processor '{HF_LLM_MODEL_ID}' loaded successfully.")
        except Exception as e:
            print(f"🔴 FATAL: Could not load Hugging Face Model/Processor '{HF_LLM_MODEL_ID}'. Error: {e}")
            print("   Ensure model ID is correct, you have internet access from compute node (for first download),")
            print("   and sufficient GPU VRAM is allocated in your Slurm job.")
            raise e

# --- Initialize Embedding Function ---
def get_embedding_function():
    global embedding_model_instance
    if embedding_model_instance is None:
        try:
            embedding_model_instance = StellaLangChainEmbeddings() # Uses global STELLA configs
            # No print here, it's printed inside StellaLangChainEmbeddings init
        except Exception as e:
            # Error already printed in StellaLangChainEmbeddings, re-raise to stop pipeline
            raise e # Or handle more gracefully if you want the script to attempt to continue
    return embedding_model_instance

# --- Phase 1: Building or Loading the Knowledge Base ---
def load_or_create_knowledge_base(current_embedding_function):
    if not current_embedding_function:
        print("🔴 Embedding function not initialized. Cannot build knowledge base.")
        return None

    if not os.path.exists(CHROMA_PERSIST_DIR):
        os.makedirs(CHROMA_PERSIST_DIR)
        print(f"Created Chroma persist directory: {CHROMA_PERSIST_DIR}")

    is_existing_db = False
    if os.path.exists(CHROMA_PERSIST_DIR) and len(os.listdir(CHROMA_PERSIST_DIR)) > 0:
        expected_files = ["chroma.sqlite3"]
        if any(f in os.listdir(CHROMA_PERSIST_DIR) for f in expected_files):
            is_existing_db = True
            print(f"Potential existing vector store found in: {CHROMA_PERSIST_DIR}")
        else:
            print(f"Directory {CHROMA_PERSIST_DIR} exists but doesn't look like a Chroma DB. Will attempt to create new.")

    vector_store = None
    if is_existing_db:
        print(f"Attempting to load existing vector store from: {CHROMA_PERSIST_DIR} with collection: {COLLECTION_NAME}")
        try:
            vector_store = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=current_embedding_function,
                collection_name=COLLECTION_NAME
            )
            if vector_store._collection.count() == 0:
                print(f"Warning: Loaded vector store, but collection '{COLLECTION_NAME}' is empty. Will rebuild.")
                is_existing_db = False
            else:
                print(f"Vector store loaded. Collection '{COLLECTION_NAME}' has {vector_store._collection.count()} documents.")
        except Exception as e:
            print(f"🔴 Error loading existing vector store: {e}. Will attempt to create a new one.")
            is_existing_db = False

    if not is_existing_db:
        print(f"Creating new vector store with {EMBEDDING_MODEL_NAME_FOR_PRINT} and COSINE similarity...")
        print(f"Loading documents from: {DOCUMENT_DIR}")
        loader_kwargs = {'encoding': 'utf-8'}
        loader = DirectoryLoader(
            DOCUMENT_DIR, glob="**/*.txt", loader_cls=TextLoader,
            loader_kwargs=loader_kwargs, show_progress=True, use_multithreading=False
        )
        try:
            documents = loader.load()
        except Exception as e:
            print(f"🔴 An error occurred during document loading: {e}")
            return None
        if not documents: print("No documents found."); return None
        print(f"Loaded {len(documents)} document(s).")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=100, length_function=len
        )
        doc_chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(doc_chunks)} chunks.")
        if not doc_chunks: print("No chunks created."); return None

        print(f"Creating vector store using {EMBEDDING_MODEL_NAME_FOR_PRINT} embeddings...")
        try:
            vector_store = Chroma.from_documents(
                documents=doc_chunks,
                embedding=current_embedding_function,
                collection_name=COLLECTION_NAME,
                persist_directory=CHROMA_PERSIST_DIR,
                collection_metadata={"hnsw:space": "cosine"}
            )
            print(f"Vector store created and persisted at: {CHROMA_PERSIST_DIR}")
            print(f"Collection '{COLLECTION_NAME}' has {vector_store._collection.count()} documents.")
        except Exception as e:
            print(f"🔴🔴🔴 FATAL ERROR during Chroma.from_documents: {e}")
            return None
    return vector_store

# --- Phase 2: Retrieval and Generation ---
def find_relevant_chunks_from_vectorstore(vector_store, query_text, top_n=15): # Using top_n=15
    if not vector_store:
        print("🔴 Vector store is not initialized.")
        return []
    print(f"\nSearching for relevant chunks for query: '{query_text[:60]}...'")
    results_with_scores = vector_store.similarity_search_with_relevance_scores(query_text, k=top_n)
    relevant_chunks = [{'text': doc.page_content, 'source': doc.metadata.get('source', 'Unknown'), 'score': score} for doc, score in results_with_scores]
    return relevant_chunks

def generate_response_with_llm(question_for_llm: str, context_str_for_llm: str):
    global llm_model, llm_processor # Use llm_processor here
    if llm_model is None or llm_processor is None:
        print("🔴 LLM model and processor not loaded. This should have been done at script start.")
        return "Error: LLM model/processor was not loaded."

    # Gemma 3 model card shows messages as a list of dictionaries.
    # For text-only RAG, we don't have an image.
    # Your enhanced prompt logic:
    system_message_content = """You are a precision-focused Q&A engine. Your sole task is to accurately answer the user's question based *only and entirely* on the provided context.
Key Instructions:
1.  **Analyze Thoroughly:** Scrutinize all provided context snippets to locate relevant information for the question.
2.  **Formulating Your Answer:**
    * **Direct Extraction:** If the answer is explicitly stated, extract it precisely.
    * **Synthesis & Inference:** If the answer is not explicit but can be logically derived or inferred by combining information from one or more context snippets, or by reasoning *directly* from the provided text, then formulate the answer. Ensure inferences are strongly supported.
    * **Completeness:** If multiple pieces of information from the context contribute to a comprehensive answer, synthesize them.
3.  **Strict Contextual Grounding (Crucial):**
    * Your answer MUST be based *solely* on the information found within the provided context snippets.
    * Do NOT use any external knowledge, prior training, or assumptions outside of what is explicitly given in the context.
4.  **Handling Unanswerable Questions:**
    * If, after careful and thorough analysis, you determine that the information required to answer the question is *definitively not present* or cannot be logically inferred from the provided context, you MUST respond with the exact phrase: "I don't know based on the provided context".
    * Do not add any apologies, explanations, or additional phrases to this specific response."""

    user_message_content = f"""**Provided Context:**
---
{context_str_for_llm}
---
**Question:** {question_for_llm}"""

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_message_content}]}, # Gemma3 expects content as a list
        {"role": "user", "content": [{"type": "text", "text": user_message_content}]}    # Gemma3 expects content as a list
    ]
    
    print(f"\n--- Preparing prompt for Hugging Face LLM ({HF_LLM_MODEL_ID}) using apply_chat_template ---")
    try:
        # The processor (stored in llm_processor) handles applying the chat template
        inputs = llm_processor.apply_chat_template( # Use llm_processor here
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(llm_model.device) # Move inputs to the same device as the model
    except Exception as e:
        print(f"🔴 Error applying chat template or moving inputs to device: {e}")
        return "Error during prompt preparation for the LLM."

    input_len = inputs["input_ids"].shape[-1]

    print(f"--- Generating response with {HF_LLM_MODEL_ID} (input length: {input_len} tokens) ---")
    try:
        with torch.inference_mode():
            generation_outputs = llm_model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False, # For more deterministic RAG answers
                # Refer to Gemma 3 model card or transformers docs for other generation params
                # pad_token_id might be needed if your processor doesn't set it and model complains.
                # Often processor.tokenizer.eos_token_id can be used if pad_token_id is None
            )
        generated_ids = generation_outputs[0][input_len:]
        generated_text = llm_processor.decode(generated_ids, skip_special_tokens=True) # Use llm_processor to decode
        
        return generated_text.strip()
    except Exception as e:
        print(f"🔴 Error generating response from Hugging Face LLM: {e}")
        import traceback
        traceback.print_exc()
        return "Sorry, I encountered an error while trying to generate a response from the LLM."

# --- Main RAG Pipeline ---
def run_rag_pipeline():
    print("--- Initializing Embedding Model ---")
    active_embedding_func = get_embedding_function()
    if not active_embedding_func:
        print("🔴 Failed to initialize embedding function. Exiting.")
        return
    
    print("\n--- Initializing LLM and Processor ---")
    try:
        load_hf_llm_and_processor() # Corrected function name
    except Exception as e:
        print(f"�� Failed to load Hugging Face LLM/Processor. Cannot proceed. Error: {e}")
        return
    
    if not llm_model or not llm_processor: # Check globals after loading
        print("🔴 LLM model or processor is not available after attempting to load. Exiting.")
        return

    print("\n--- Step 1: Initializing/Loading Knowledge Base ---")
    print(f"--- Using {EMBEDDING_CHOICE.upper()} embeddings ({EMBEDDING_MODEL_NAME_FOR_PRINT}) ---")
    vector_store = load_or_create_knowledge_base(active_embedding_func)
    if not vector_store:
        print("🔴 Failed to initialize knowledge base. Exiting.")
        return

    print("\n--- Step 2: Ready to answer questions ---")
    print(f"--- Using LLM: {HF_LLM_MODEL_ID} ---")
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

        relevant_chunks = find_relevant_chunks_from_vectorstore(vector_store, user_query, top_n=15)

        if not relevant_chunks:
            print("No relevant context found for your query.")
            final_answer = "I couldn't find any relevant information in my current knowledge base to answer your question."
        else:
            print(f"Found {len(relevant_chunks)} relevant chunk(s):")
            context_for_llm_parts = []
            for i, chunk_info in enumerate(relevant_chunks):
                source_filename = os.path.basename(chunk_info['source'])
                score_display = f"{chunk_info['score']:.4f}" if isinstance(chunk_info['score'], float) else "N/A"
                print(f"  {i+1}. Source: {source_filename}, Relevance Score: {score_display}")
                print(f"     Content Snippet: {chunk_info['text'][:150].strip()}...")
                context_for_llm_parts.append(f"Source: {source_filename}\nContent: {chunk_info['text']}")
            
            context_str_for_llm = "\n\n---\n\n".join(context_for_llm_parts)

            print("Generating answer...")
            final_answer = generate_response_with_llm(user_query, context_str_for_llm)

        print("\n--- Answer ---")
        print(final_answer)

if __name__ == '__main__':
    print("Script starting via __main__...")
    try:
        run_rag_pipeline()
        print("Script finished execution successfully.")
    except Exception as e_main:
        print(f"🔴 An unexpected error occurred in the main execution block: {e_main}")
        import traceback
        traceback.print_exc()

