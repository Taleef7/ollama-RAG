# test_gpu.py

import torch

import os # Import os to check environment variables if needed



print(f"Torch version: {torch.__version__}")



# Optional: Check CUDA_VISIBLE_DEVICES if you need to debug GPU selection

# print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")



if torch.cuda.is_available():

    print(f"CUDA is available. Number of GPUs: {torch.cuda.device_count()}")

    # Try to get the name of the first GPU (index 0)

    try:

        print(f"Current GPU name: {torch.cuda.get_device_name(0)}")

        device = torch.device("cuda")

        # Simple tensor operation on GPU

        print("Performing a simple tensor operation on GPU...")

        x = torch.tensor([1.0, 2.0, 3.0], device=device)

        y = x * 2

        print(f"Tensor on GPU: {y}")

        print(f"Tensor device: {y.device}")

        print("GPU tensor test successful.")

    except Exception as e:

        print(f"Error during GPU operations: {e}")

        print("Falling back to CPU for Stella test if GPU operation failed.")

        device = torch.device("cpu") # Fallback for Stella if GPU part had issues

else:

    print("CUDA is not available. Running on CPU.")

    device = torch.device("cpu")



# Test sentence-transformers with Stella on the chosen device

try:

    from sentence_transformers import SentenceTransformer

    print("\nAttempting to load Stella model...")



    # Use the 'device' determined above (cuda if available and working, else cpu)

    stella_device_to_test = device.type # Will be 'cuda' or 'cpu'

    print(f"Testing Stella on: {stella_device_to_test}")



    model_init_kwargs_test = {"trust_remote_code": True}

    # This config_kwargs is mainly if xformers is not installed;

    # since you have it, it might not be strictly necessary, but good for robustness.

    if stella_device_to_test == "cpu":

        model_init_kwargs_test["config_kwargs"] = {

            "use_memory_efficient_attention": False,

            "unpad_inputs": False

        }



    stella_model = SentenceTransformer(

        "NovaSearch/stella_en_400M_v5", 

        device=stella_device_to_test,

        **model_init_kwargs_test

    )

    print("Stella model loaded successfully.")

    test_sentence = ["This is a test sentence from Gilbreth."]

    embedding = stella_model.encode(test_sentence)

    print(f"Stella embedding shape: {embedding.shape}")

    print("Stella embedding test successful.")

except Exception as e:

    print(f"Error testing Stella model: {e}")



print("\nTest script finished.")
