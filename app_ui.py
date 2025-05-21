import streamlit as st
import os
import time # Only for dummy placeholders if needed, can be removed

# Import the refactored functions and necessary constants from your run_rag.py
try:
    from run_rag import (
        get_embedding_model_instance,
        get_llm_and_processor,
        load_or_create_knowledge_base,
        find_relevant_chunks_from_vectorstore,
        generate_response_with_llm,
        HF_LLM_MODEL_ID, # Import constants
        EMBEDDING_MODEL_NAME_FOR_PRINT
    )
    RAG_FUNCTIONS_LOADED = True
except ImportError as e:
    st.error(f"Failed to import functions from run_rag.py. Ensure it's in the same directory. Error: {e}")
    RAG_FUNCTIONS_LOADED = False

# --- Streamlit Caching for Models and Data ---

@st.cache_resource(show_spinner="Loading Embedding Model...")
def cached_get_embedding_model():
    print("STREAMLIT CACHE: Getting embedding model instance...")
    return get_embedding_model_instance()

@st.cache_resource(show_spinner="Loading LLM and Processor (this may take a while on first startup)...")
def cached_get_llm_and_processor():
    print("STREAMLIT CACHE: Getting LLM and Processor instances...")
    return get_llm_and_processor()

@st.cache_resource(show_spinner="Loading/Creating Knowledge Base...")
def cached_get_vector_store(_embedding_model): # Pass embedding model to ensure cache invalidation if it changes
    print("STREAMLIT CACHE: Getting vector store instance...")
    # _embedding_model argument is used to help Streamlit's caching understand dependencies.
    # The actual embedding_model used by load_or_create_knowledge_base is the one returned by cached_get_embedding_model()
    return load_or_create_knowledge_base(_embedding_model)


# --- Streamlit App Layout & Logic ---
def main_ui():
    st.set_page_config(page_title="RAG System UI - Gilbreth", layout="wide")
    st.title("📚 Query Your Documents with RAG")
    
    if not RAG_FUNCTIONS_LOADED:
        st.error("Core RAG functions could not be loaded. Please check the terminal for errors from `run_rag.py` imports.")
        return

    # Load models and vector store using cached functions
    embedding_model = cached_get_embedding_model()
    llm_model, llm_processor = cached_get_llm_and_processor()
    vector_store = cached_get_vector_store(embedding_model) # Pass the loaded embedding model

    if not embedding_model or not llm_model or not llm_processor or not vector_store:
        st.error("Failed to initialize one or more RAG components. Check terminal logs from the Streamlit process.")
        return
    
    st.success(f"RAG System Ready! Embeddings: {EMBEDDING_MODEL_NAME_FOR_PRINT}, LLM: {HF_LLM_MODEL_ID}")

    # --- UI Elements ---
    st.sidebar.header("Controls")
    top_n_chunks = st.sidebar.slider("Number of chunks to retrieve for context:", min_value=1, max_value=25, value=15, step=1)

    st.header("Ask a Question")
    # Use session state to keep the input field persistent if desired, or clear on submit
    if 'user_query' not in st.session_state:
        st.session_state.user_query = ""

    user_query = st.text_input("Enter your question here:", value=st.session_state.user_query, key="query_widget")
    
    if st.button("Get Answer", key="get_answer_button"):
        st.session_state.user_query = user_query # Store current query
        if user_query:
            with st.spinner("Searching for relevant documents..."):
                retrieved_chunks_with_scores = find_relevant_chunks_from_vectorstore(vector_store, user_query, top_n=top_n_chunks)

            st.subheader("🔎 Retrieved Context Snippets:")
            if retrieved_chunks_with_scores:
                context_for_llm_parts = []
                for i, (doc, score) in enumerate(retrieved_chunks_with_scores):
                    source_filename = os.path.basename(doc.metadata.get('source', 'Unknown'))
                    with st.expander(f"Chunk {i+1} | Source: {source_filename} | Score: {score:.4f} (Relevance)"):
                        st.markdown(f"```text\n{doc.page_content}\n```") # Use markdown for better text display
                    context_for_llm_parts.append(f"Source: {source_filename}\nContent: {doc.page_content}")
                
                context_str_for_llm = "\n\n---\n\n".join(context_for_llm_parts)

                st.subheader("🤖 Generated Answer:")
                with st.spinner(f"Generating answer with {HF_LLM_MODEL_ID}..."):
                    # Call the actual generate_response_with_llm from run_rag.py
                    final_answer = generate_response_with_llm(user_query, context_str_for_llm, llm_model, llm_processor)
                st.markdown(final_answer) # Use markdown for potentially formatted LLM output
            else:
                st.warning("No relevant chunks found for your query. The LLM will not be called.")
        else:
            st.warning("Please enter a question.")

    st.sidebar.markdown("---")
    st.sidebar.info(f"Embedding Model: {EMBEDDING_MODEL_NAME_FOR_PRINT}\n\nLLM: {HF_LLM_MODEL_ID}\n\nChunks for Context: {top_n_chunks}")

if __name__ == '__main__':
    main_ui()
