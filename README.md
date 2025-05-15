# ollama-RAG Project

This project is an exploration into building a Retrieval Augmented Generation (RAG) system using locally run Large Language Models (LLMs) via Ollama, and experimenting with different embedding models. The goal is to create a pipeline that can answer questions based on a provided set of documents.

This project is part of an internship focused on "Attention Tensor-Based Text Indexing and Searching."

## Current Progress (as of May 15, 2025 - Gemini Embedding Attempt)

* **Local LLM Setup:** Continued use of Ollama for local LLM execution. Upgraded and tested with `gemma3:4b-it-qat` as the generative model.
* **Embedding Model Experimentation - Google Gemini Embeddings:**
    * Successfully set up the `google-generativeai` Python SDK and configured API key access.
    * Developed a custom LangChain `Embeddings` wrapper (`GeminiLangChainEmbeddings`) to integrate Google's experimental `gemini-embedding-exp-03-07` model.
        * This wrapper implements specific `task_type` parameters for document retrieval (`RETRIEVAL_DOCUMENT`) and query embedding (`RETRIEVAL_QUERY`).
        * Implemented batching and delay logic within the wrapper to manage API calls.
    * **Testing with Gemini Embeddings:**
        * Successfully embedded a small number of chunks (e.g., 1-3 chunks from short documents) without hitting rate limits, demonstrating functional API integration.
        * Encountered `429 Resource has been exhausted (e.g. check quota)` errors when attempting to embed a larger number of chunks (e.g., ~750 chunks from 4 documents, including longer texts) even with aggressive batching (e.g., 10 texts per batch) and significant delays (e.g., 15-25s initial, 13-20s subsequent). This indicates very strict rate limits for the experimental `gemini-embedding-exp-03-07` model on the Free Tier.
        * The RAG pipeline (using Gemini embeddings for retrieval and `gemma3:4b-it-qat` for generation) was functional for the successfully embedded small dataset, showing promising retrieval relevance.
* **Previous State (mxbai-embed-large via Ollama):**
    * Successfully implemented a full RAG pipeline using `mxbai-embed-large` for embeddings (served via Ollama) and `gemma3:1b-it-qat` (and later `gemma3:4b-it-qat`) for generation.
    * Utilized ChromaDB with cosine similarity for persistent vector storage.
    * Addressed `mxbai-embed-large` specific query prefix requirements.
    * Corrected LangChain deprecation warnings for `OllamaEmbeddings` and `Chroma`.
    * Observed improved answer quality with `gemma3:4b-it-qat` for some questions (e.g., "bacteria killed Martians") but also some regressions or persistent "I don't know" responses, highlighting the ongoing importance of retrieval quality.

## Technologies Used (Reflecting Gemini Attempt)

* **Python 3.x**
* **Ollama:** For running local LLMs (`gemma3:4b-it-qat`).
* **`ollama` Python library:** For direct LLM chat interaction.
* **Google Generative AI SDK (`google-generativeai`):** For using Gemini Embedding API.
* **Python DotEnv (`python-dotenv`):** For managing API keys.
* **LangChain (`langchain`, `langchain_community`, `langchain_ollama`, `langchain_chroma`):**
    * `DirectoryLoader` and `TextLoader`.
    * `RecursiveCharacterTextSplitter`.
    * Custom `GeminiLangChainEmbeddings` wrapper.
    * `Chroma` vector store.
* **ChromaDB:** As the local vector database.
* **NumPy**.
* **TikToken**.

## Next Steps

1.  **Revert to Local Embedding Model:** Due to persistent rate limit issues with the experimental Gemini Embedding API's Free Tier, the immediate next step is to revert to a fully local embedding model (e.g., `mxbai-embed-large` via Ollama or experimenting with other Hugging Face models like `NovaSearch/stella_en_400M_v5` via `sentence-transformers`).
2.  **Continue RAG Pipeline Refinement:**
    * Further experiment with `chunk_size`, `chunk_overlap`, and `top_n` for the chosen local embedding model.
    * Focus on improving retrieval for questions that are still challenging.
3.  **Develop a User Interface:** Create a UI using Streamlit.
4.  **Evaluation & MIMIC-III Preparation.**

## Setup and Running (Reflecting Gemini Attempt - Note on Rate Limits)

1.  Ensure Ollama is installed and running (for the LLM).
2.  (For Gemini Embedding Attempt) Ensure Google Cloud Project is set up, "Generative Language API" is enabled, and `GEMINI_API_KEY` is in a `.env` file. **Note: Experimental Gemini embedding model has very strict Free Tier rate limits.**
3.  Pull the required Ollama LLM:
    ```bash
    ollama pull gemma3:4b-it-qat
    ```
4.  Clone this repository.
5.  Create a Python virtual environment and install dependencies from `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
6.  Create a `data/` directory and add source `.txt` documents.
7.  Run the RAG script:
    ```bash
    python run_rag.py
    ```
    If using Gemini embeddings for the first time, building the ChromaDB will be slow and may hit rate limits for large datasets. For local Ollama embeddings, it will depend on local machine performance.

## Credit and AI tools used
This project makes use of the help of Gemini 2.5 for code generation, debugging, and documentation. The initial idea and structure were inspired by the LangChain documentation and examples, but the implementation is original and tailored to the specific requirements of this project.