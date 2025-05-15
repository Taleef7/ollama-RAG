# ollama-RAG Project

This project is an exploration into building a Retrieval Augmented Generation (RAG) system using locally run Large Language Models (LLMs) via Ollama. The goal is to create a pipeline that can answer questions based on a provided set of documents by retrieving relevant information and then using an LLM to generate a contextual answer.

This project is part of an internship focused on "Attention Tensor-Based Text Indexing and Searching."

## Current Progress (as of May 15, 2025)

* **Initial Ollama Setup:** Successfully set up Ollama for local LLM (`gemma3:1b-it-qat`) and embedding model (`mxbai-embed-large`) execution.
* **Core RAG Pipeline Implemented (`run_rag.py`):**
    * **Document Loading:** Loads `.txt` documents from a local `data/` directory using LangChain's `DirectoryLoader` with UTF-8 encoding.
    * **Chunking:** Implemented text chunking using LangChain's `RecursiveCharacterTextSplitter` (e.g., `chunk_size=700`, `chunk_overlap=150`), successfully processing longer literary texts ("War of the Worlds," "Dante's Inferno") into numerous chunks (e.g., 2154 chunks from 5 documents).
    * **Embedding Generation:** Utilizes `mxbai-embed-large` via `langchain_ollama.OllamaEmbeddings`.
    * **Vector Store:** Integrated **ChromaDB** as a persistent local vector store using **cosine similarity** (`collection_metadata={"hnsw:space": "cosine"}`). The vector store is created if it doesn't exist or loaded if it does.
    * **Retrieval:**
        * Embeds user queries using `mxbai-embed-large` with the model-specific prefix ("Represent this sentence for searching relevant passages: ...").
        * Performs similarity search against ChromaDB using `similarity_search_with_relevance_scores` to find the top N relevant chunks (tested with `top_n=4`).
    * **Augmented Generation:**
        * Constructs an augmented prompt with enhanced instructions for the LLM, including the user's query and the retrieved context.
        * Uses `gemma3:1b-it-qat` via the `ollama` Python library to generate answers.
* **Testing & Observations:**
    * Successfully processed and indexed longer documents into a significant number of chunks.
    * Improved LLM prompt led to better answer extraction from varied context types (e.g., table of contents).
    * Observed improvements in some answers with `top_n=4` and refined chunking (e.g., description of Martian fighting machines, color of Heat-Ray, main character survival).
    * Persistent challenges remain for some queries where the most specific information is not in the top retrieved chunks, or where LLM interpretation of the provided snippets is difficult. Some answers correctly identify "I don't know," while others show misinterpretation.
* **Addressed Deprecation Warnings:** Updated imports for `OllamaEmbeddings` (to `langchain_ollama`) and `Chroma` (to `langchain_chroma`).

## Technologies Used So Far

* **Python 3.x**
* **Ollama:** For running local LLMs (`gemma3:1b-it-qat`) and embedding models (`mxbai-embed-large`).
* **`ollama` Python library:** For direct interaction with the Ollama API (chat).
* **LangChain (`langchain`, `langchain_community`, `langchain_ollama`, `langchain_chroma`):**
    * `DirectoryLoader` and `TextLoader`.
    * `RecursiveCharacterTextSplitter`.
    * `OllamaEmbeddings`.
    * `Chroma` vector store.
* **ChromaDB:** As the local vector database.
* **NumPy:** (Primarily for initial similarity tests, less direct use now Chroma handles it).
* **TikToken:** (Dependency for LangChain text splitters).

## Next Steps

1.  **Experiment with different Embedding and LLM Models:** Explore alternatives to `mxbai-embed-large` and `gemma3:1b-it-qat` to see impact on retrieval and generation quality.
2.  **Further Refine Retrieval:**
    * Continue experimenting with `chunk_size`, `chunk_overlap`, and `top_n`.
    * Investigate methods to improve the ranking of retrieved chunks if highly relevant information is present but not in the top results.
3.  **Develop a User Interface:** Create a UI using Streamlit.
4.  **Evaluation:** Define and implement metrics for RAG performance.
5.  **Prepare for MIMIC-III:** Continue efforts to gain access and plan for preprocessing this dataset.

## Setup and Running

1.  Ensure Ollama is installed and running.
2.  Pull the required models (or the new ones you plan to test):
    ```bash
    ollama pull gemma3:1b-it-qat
    ollama pull mxbai-embed-large
    ```
3.  Clone this repository.
4.  Create a Python virtual environment and install dependencies from `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
5.  Create a `data/` directory in the project root and add your `.txt` source documents.
6.  Run the RAG script:
    ```bash
    python run_rag.py
    ```
    The first run (or after deleting `chroma_db_ollama/`) will build the ChromaDB. Subsequent runs will load from it.

## Credit and AI tools used
This project makes use of the help of Gemini 2.5 for code generation, debugging, and documentation. The initial idea and structure were inspired by the LangChain documentation and examples, but the implementation is original and tailored to the specific requirements of this project.