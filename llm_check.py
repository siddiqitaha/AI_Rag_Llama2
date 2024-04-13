import os

def check_and_pull_llama2():
    # Define the file path
    file_path = "/Users/username/.ollama/models/manifests/registry.ollama.ai/library/llama2/latest"
    
    # Check if the file exists
    if not os.path.exists(file_path):
        # File does not exist, so pull the model
        os.system("ollama pull llama2")
        print("Pulled llama2 model.")
    else:
        print("llama2 model already exists.")


def check_and_pull_nomic_embed_text():
    # Define the file path
    file_path = "/Users/username/.ollama/models/manifests/registry.ollama.ai/library/nomic-embed-text/latest"
    
    # Check if the file exists
    if not os.path.exists(file_path):
        # File does not exist, so pull the model
        os.system("ollama pull nomic-embed-text")
        print("Pulled nomic-embed-text model.")
    else:
        print("nomic-embed-text model already exists.")