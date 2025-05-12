import ollama
# import json # No longer needed for basic printing

# Ensure the models are available via Ollama service
MODEL_NAME = 'gemma3:1b-it-qat'
EMBEDDING_MODEL_NAME = 'mxbai-embed-large' # Make sure you pulled this model too

def generate_text(prompt_text):
    """Generates text using the specified model."""
    print(f"--- Generating text for prompt: '{prompt_text}' ---")
    try:
        # Simple chat example
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt_text}]
        )
        # The response object is a dictionary-like object
        print("Chat Response Object (raw print):")
        print(response) # Print the raw response object for inspection if needed
        print("\nAssistant's Message:")
        # Correctly accessing the content
        assistant_message = response['message']['content']
        print(assistant_message)
        return assistant_message # Return just the content
    except Exception as e:
        print(f"Error during text generation: {e}")
        # You might want to print the raw response even on error for debugging
        try:
            print("Raw response on error:", response)
        except NameError: # response might not be defined if error happened early
            pass
        except Exception as E_print:
            print("Could not print raw response on error", E_print)
        return None

def generate_embedding(text_to_embed):
    """Generates embeddings for the given text."""
    print(f"--- Generating embedding for text: '{text_to_embed}' ---")
    try:
        # Generate embeddings
        embedding_response = ollama.embeddings(
            model=EMBEDDING_MODEL_NAME,
            prompt=text_to_embed
            # Note: For mxbai-embed-large retrieval queries, the prompt should be:
            # prompt=f"Represent this sentence for searching relevant passages: {text_to_embed}"
            # For embedding documents, just use the text itself.
        )
        # The response object is a dictionary containing the embedding vector
        print("Embedding Response:")
        # print(embedding_response) # Print the whole response if needed
        actual_embedding = embedding_response['embedding']
        print(f"Embedding vector length: {len(actual_embedding)}")
        print(f"First few elements: {actual_embedding[:5]}...") # Print first 5 elements
        return actual_embedding
    except Exception as e:
        print(f"Error during embedding generation: {e}")
        return None

# --- Example Usage ---
if __name__ == "__main__":
    # Example 1: Generate Text
    prompt = "Why is the sky blue?"
    generated_response = generate_text(prompt)

    print("\n" + "="*30 + "\n")

    # Example 2: Generate Embedding
    text = "The quick brown fox jumps over the lazy dog."
    embedding_vector = generate_embedding(text)

    print("\n" + "="*30 + "\n")

    # Example 3: Generate batch embeddings
    try:
        print("--- Generating batch embeddings ---")
        batch_texts = ["Text one", "Second piece of text"]
        # Using ollama.embed as shown in some library examples
        # Note: mxbai doesn't need a special prompt for documents being embedded,
        # only for the *query* text when doing retrieval search.
        batch_response = ollama.embed(
             model=EMBEDDING_MODEL_NAME,
             input=batch_texts
        )
        print("Batch Embedding Response Object (raw print):")
        print(batch_response) # Print the raw response object if needed

        # --- Access the embeddings correctly ---
        if 'embeddings' in batch_response:
            actual_embeddings_list = batch_response['embeddings']
            print(f"\nGenerated {len(actual_embeddings_list)} embeddings.")
            if len(actual_embeddings_list) > 0:
                 print(f"Embedding 1 length: {len(actual_embeddings_list[0])}")
                 print(f"Embedding 1 first few elements: {actual_embeddings_list[0][:5]}...")
        else:
            print("Could not find 'embeddings' key in the batch response.")

    except AttributeError:
         print("\nBatch embedding with ollama.embed not available or structured differently in this version/context.")
         print("Consider trying ollama.embeddings with a list input if supported, or loop generate_embedding.")
    except Exception as e:
         print(f"\nError during batch embedding: {e}")
         # Print raw response on error if possible
         try:
             print("Raw batch response on error:", batch_response)
         except NameError:
             pass
         except Exception as E_print:
             print("Could not print raw batch response on error", E_print)