"""
Embedding generation module.
"""
from typing import Any


def load_embedding_model(model_name: str, device: str | None = None) -> Any:
    """
    Loads and returns a SentenceTransformer model, using a GPU if available.

    This function should be called once to initialize the model.

    Args:
        model_name: The name of the model to load (e.g., 'all-MiniLM-L6-v2').
        device: Optional device string. Use "auto" or None to prefer CUDA when available.

    Returns:
        An instance of the SentenceTransformer model, or None if an error occurs.
    """
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        print(
            "Error loading embedding dependencies. Install dependencies with "
            "`pip install -r requirements.txt`."
        )
        print(f"Missing dependency detail: {exc}")
        return None

    # Prefer the available CUDA device on DGX/local GPU systems, but allow config overrides.
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        print(f"Loading embedding model: {model_name}...")
        model = SentenceTransformer(model_name, device=device)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None


def get_embedding(text: str, model: Any) -> list[float]:
    """
    Generates a vector embedding for the given text using a pre-loaded model.

    Args:
        text: The input string to embed.
        model: The pre-loaded SentenceTransformer model instance.

    Returns:
        A list of floats representing the vector embedding, or an empty list on error.
    """
    if model is None or not hasattr(model, "encode"):
        print("Error: A valid embedding model with an encode method must be provided.")
        return []

    try:
        text_to_embed = text.replace("\n", " ")
        if not text_to_embed:
            return []

        embedding_array = model.encode(text_to_embed)
        return embedding_array.tolist()

    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []
