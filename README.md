# Advanced RAG System on Purdue Gilbreth Cluster

This project implements a Retrieval Augmented Generation (RAG) pipeline designed to run on Purdue's Gilbreth RCAC cluster, leveraging powerful Hugging Face models for embeddings and language generation, with a Streamlit user interface.

**Last Updated:** May 21, 2025

## Core Architecture
* **Embedding Model:** `Linq-AI-Research/Linq-Embed-Mistral` (loaded via `sentence-transformers`) for generating high-quality embeddings for documents and queries.
* **Language Model (LLM):** `google/gemma-3-12b-it` (12B parameters, instruction-tuned) loaded directly via the Hugging Face `transformers` library with `attn_implementation="eager"` for robust GPU execution.
* **Vector Store:** ChromaDB for storing and retrieving document embeddings, using cosine similarity.
* **Chunking:** `RecursiveCharacterTextSplitter` with `CHUNK_SIZE=2048` and `CHUNK_OVERLAP=200` to align with the capabilities of `Linq-Embed-Mistral`.
* **User Interface:** Streamlit application (`app_ui.py`) for interactive querying.
* **Execution Environment:** Purdue Gilbreth RCAC Cluster, utilizing Slurm for job scheduling and GPUs (e.g., NVIDIA A100) for model acceleration.

## Key Features & Progress
* Successful setup and execution on Gilbreth compute nodes.
* Integration of large, state-of-the-art embedding (Linq-Embed-Mistral) and LLM (Gemma 3 12B-IT) models.
* Dynamic retrieval of `top_n` context chunks (currently tested up to `n=15`).
* Resolution of HPC-specific challenges:
    * SQLite version compatibility for ChromaDB (using `pysqlite3-binary`).
    * Hugging Face gated model access and token authentication.
    * Installation of `accelerate` for large model support (`device_map="auto"`).
    * Home directory disk quota management by redirecting caches (`HF_HOME`, etc.) to scratch space.
    * PyTorch attention mechanism alignment errors (`attn_implementation="eager"`).
* Streamlit UI for user interaction.
* Knowledge base includes: `waroftheworlds.txt`, `dantes_inferno.txt`, `a_tale_of_two_cities.txt`, and shorter documents `doc1.txt` (Eiffel Tower), `doc2.txt` (Photosynthesis), `doc3.txt` (Python).

## How to Run (on Gilbreth)

1.  **Clone the Repository:**
    ```bash
    git clone <your_repo_url>
    cd <your_repo_name>
    ```
2.  **Set up Python Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
3.  **Configure Hugging Face Authentication:**
    * Ensure you have access to gated models like `google/gemma-3-12b-it` on the Hugging Face website.
    * Log in using the CLI (ideally within an `sinteractive` session before first model download):
        ```bash
        huggingface-cli login
        ```
    * Ensure `HF_HOME` environment variable is set in your `~/.bashrc` to point to your scratch space for caching models, e.g.:
        ```bash
        export HF_HOME="/scratch/gilbreth/your_username/huggingface_cache"
        ```
        And create this directory. Source `~/.bashrc` or re-login.

4.  **Prepare Data:**
    * Place your `.txt` document files into the `data/` subdirectory.

5.  **Run the Streamlit UI (Interactive via `sinteractive`):**
    * Request an interactive session with a powerful GPU (e.g., A100 with >=40GB VRAM):
        ```bash
        sinteractive --account=<your_account> --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=128G --time=02:00:00 --gres=gpu:1 --constraint="A100-80GB" 
        # Adjust account, constraint, mem, time as needed
        ```
    * Once on the compute node:
        ```bash
        cd /path/to/your/project/ollama-RAG 
        source venv/bin/activate
        streamlit run app_ui.py --server.port <your_chosen_port, e.g., 8501>
        ```
    * Set up SSH port forwarding from your local machine to access the Streamlit app in your browser (e.g., `ssh -L <local_port>:<compute_node_hostname>:<remote_port> your_username@gilbreth.rcac.purdue.edu`).
    * Alternatively, use the "Compute Node Desktop" app via Open OnDemand to run Streamlit and a browser within the VNC session.

6.  **Run CLI Version (for debugging/batch, also via `sinteractive` or `sbatch`):**
    ```bash
    # Within an sinteractive session with resources:
    python run_rag.py 
    ```
    For `sbatch`, you'll need a submission script (see `test_gpu.slurm` for an example structure) and potentially modify `run_rag.py` to not expect interactive input.

## Current Challenges / Next Steps
* Investigating and improving retrieval precision for queries targeting specific smaller documents within the larger mixed corpus (e.g., `doc3.txt` for Python-related questions).
* Exploring advanced RAG techniques like re-ranking if necessary.
* Systematic evaluation of RAG performance.
* Potential exploration of alternative optimized attention mechanisms (e.g., "sdpa", "flash_attention_2") for the LLM if "eager" proves too slow for extensive use.

## Dependencies
Key dependencies are listed in `requirements.txt`. Ensure `pysqlite3-binary` and `accelerate` are included.

## Credit and AI tools used
This project makes use of the help of Gemini 2.5 for code generation, debugging, and documentation. The initial idea and structure were inspired by the LangChain documentation and examples, but the implementation is original and tailored to the specific requirements of this project.
