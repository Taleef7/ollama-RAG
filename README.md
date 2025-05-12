# ollama-RAG Project

This project is an exploration into building a Retrieval Augmented Generation (RAG) system using locally run Large Language Models (LLMs) via Ollama. The goal is to create a pipeline that can answer questions based on a provided set of documents by retrieving relevant information and then using an LLM to generate a contextual answer.

This project is part of an internship focused on "Attention Tensor-Based Text Indexing and Searching."

## Current Progress (as of May 12, 2025)

* **Initial Ollama Setup:** Successfully set up Ollama to run local LLMs and embedding models.
    * Tested text generation with `gemma3:1b-it-qat`.
    * Tested embedding generation with `mxbai-embed-large`.
* **Basic RAG Pipeline Implemented (`run_rag.py`):**
    * **Document Loading:** Loads text documents from a local `data/` directory.
    * **Chunking (Simple):** Currently treats each document as a single chunk (to be improved).
    * **Embedding Generation:** Generates embeddings for each document chunk using `mxbai-embed-large` via Ollama.
    * **In-Memory Knowledge Base:** Stores document chunks and their embeddings in a Python list.
    * **Retrieval:**
        * Embeds user queries using `mxbai-embed-large`.
        * Performs cosine similarity search against the in-memory knowledge base to find the top N relevant chunks.
    * **Augmented Generation:**
        * Constructs an augmented prompt containing the user's query and the retrieved context.
        * Uses `gemma3:1b-it-qat` via Ollama to generate a final answer based on the augmented prompt.
* **Successful Initial Tests:**
    * The pipeline correctly answers questions when relevant information is present in the sample documents.
    * The system correctly states "I don't know based on the provided context" when the information is not available in the loaded documents, demonstrating proper grounding.

## Technologies Used So Far

* **Python 3.12**
* **Ollama:** For running local LLMs (`gemma3:1b-it-qat`) and embedding models (`mxbai-embed-large`).
* **`ollama` Python library:** For interacting with the Ollama API.
* **NumPy:** For cosine similarity calculation.

## Next Steps

1.  **Improve Document Chunking:** Implement more sophisticated chunking strategies (e.g., paragraph splitting, fixed-size with overlap).
2.  **Integrate a Vector Database:** Replace the in-memory knowledge base with a persistent local vector store (e.g., ChromaDB or FAISS) for better scalability and efficiency.
3.  **Refine Retrieval and Prompting:** Experiment with the number of retrieved chunks (`top_n`) and prompt engineering techniques.
4.  **Develop a User Interface:** Create a simple UI using Streamlit.
5.  **Explore LangChain Integration:** Consider leveraging LangChain for orchestrating the RAG pipeline.
6.  **Prepare for Larger Datasets:** Scale the system to handle more complex datasets like MIMIC-III.

## Setup and Running

1.  Ensure Ollama is installed and running.
2.  Pull the required models:
    ```bash
    ollama pull gemma3:1b-it-qat
    ollama pull mxbai-embed-large
    ```
3.  Clone this repository.
4.  Create a Python virtual environment and install dependencies:
    ```bash
    pip install ollama numpy
    ```
5.  Create a `data/` directory in the project root and add your `.txt` source documents.
6.  Run the RAG script:
    ```bash
    python run_rag.py
    ```

## Credit and AI tools used
This project makes use of the help of Gemini 2.5 for code generation and debugging. The initial idea and structure were inspired by the LangChain documentation and examples, but the implementation is original and tailored to the specific requirements of this project.