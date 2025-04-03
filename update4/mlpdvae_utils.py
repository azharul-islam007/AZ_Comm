import os
import numpy as np
import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_transmission_pairs(max_pairs=1000, transmission_dir='./transmission_pairs'):
    """
    Load saved transmission pairs for self-supervised learning.

    Args:
        max_pairs: Maximum number of pairs to load
        transmission_dir: Directory containing transmission pair files

    Returns:
        List of (original, received) embedding pairs
    """
    if not os.path.exists(transmission_dir):
        logger.warning(f"Transmission pairs directory {transmission_dir} not found")
        return []

    pairs = []
    pair_files = sorted([f for f in os.listdir(transmission_dir)
                         if f.startswith('pair_') and f.endswith('.npz')])

    # Limit to max_pairs
    pair_files = pair_files[:max_pairs]

    for file in tqdm(pair_files, desc="Loading transmission pairs"):
        try:
            data = np.load(os.path.join(transmission_dir, file))
            original = data['original']
            received = data['received']
            pairs.append((original, received))
        except Exception as e:
            logger.warning(f"Error loading transmission pair {file}: {e}")

    logger.info(f"Loaded {len(pairs)} transmission pairs for self-supervised learning")
    return pairs


def evaluate_reconstruction_with_semantics(original_text, reconstructed_text, semantic_loss_fn=None):
    """
    Calculate enhanced evaluation metrics for reconstruction quality including semantic metrics.

    Args:
        original_text: Original text
        reconstructed_text: Reconstructed text
        semantic_loss_fn: Optional semantic loss function

    Returns:
        Dictionary of metrics
    """
    # Import necessary libraries
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from rouge_score import rouge_scorer
        from nltk.translate.meteor_score import meteor_score
    except ImportError:
        logger.warning("NLTK or rouge_score not available, some metrics will be skipped")

    # Check for empty reconstruction
    if not reconstructed_text or reconstructed_text.strip() == "":
        logger.warning("Empty reconstruction detected!")
        return {"BLEU": 0, "ROUGE1": 0, "ROUGEL": 0, "METEOR": 0, "SEMANTIC": 0}

    metrics = {}

    # Calculate BLEU score
    try:
        bleu = sentence_bleu([original_text.split()], reconstructed_text.split(),
                             smoothing_function=SmoothingFunction().method4)
        metrics["BLEU"] = bleu
    except Exception as e:
        logger.warning(f"BLEU calculation failed: {e}")
        metrics["BLEU"] = 0

    # Calculate ROUGE scores
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(original_text, reconstructed_text)
        metrics["ROUGE1"] = rouge_scores['rouge1'].fmeasure
        metrics["ROUGEL"] = rouge_scores['rougeL'].fmeasure
    except Exception as e:
        logger.warning(f"ROUGE calculation failed: {e}")
        metrics["ROUGE1"] = 0
        metrics["ROUGEL"] = 0

    # Calculate METEOR score
    try:
        meteor = meteor_score([original_text.split()], reconstructed_text.split())
        metrics["METEOR"] = meteor
    except Exception as e:
        logger.warning(f"METEOR calculation failed: {e}")
        metrics["METEOR"] = 0

    # Calculate semantic similarity if semantic loss function is provided
    if semantic_loss_fn is not None:
        try:
            semantic_similarity = semantic_loss_fn.calculate_semantic_similarity(
                original_text, reconstructed_text)
            metrics["SEMANTIC"] = semantic_similarity
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")
            metrics["SEMANTIC"] = 0
    else:
        # Default semantic similarity (average of BLEU and ROUGE-L)
        metrics["SEMANTIC"] = (metrics["BLEU"] + metrics["ROUGEL"]) / 2

    return metrics


# Update in mlpdvae_utils.py
def compute_embedding_similarity(emb1, emb2):
    """
    Calculate cosine similarity between embeddings with dimension handling.
    """
    # Convert numpy arrays to tensors if needed
    if isinstance(emb1, np.ndarray):
        emb1 = torch.tensor(emb1, dtype=torch.float32)
    if isinstance(emb2, np.ndarray):
        emb2 = torch.tensor(emb2, dtype=torch.float32)

    # Get current dimensions
    dim1 = emb1.shape[-1] if len(emb1.shape) > 0 else 0
    dim2 = emb2.shape[-1] if len(emb2.shape) > 0 else 0

    # Handle dimension mismatch
    if dim1 != dim2:
        target_dim = min(dim1, dim2)
        # Use existing adapt_dimensions function
        emb1 = adapt_dimensions(emb1, target_dim)
        emb2 = adapt_dimensions(emb2, target_dim)

    # Ensure both have consistent shape (batch dimension)
    emb1 = ensure_tensor_shape(emb1, expected_dim=2)
    emb2 = ensure_tensor_shape(emb2, expected_dim=2)

    # Ensure both are on the same device
    if emb1.device != emb2.device:
        emb2 = emb2.to(emb1.device)

    # Compute cosine similarity
    try:
        cos_sim = F.cosine_similarity(emb1, emb2, dim=1)
        # Return the mean as a scalar
        return cos_sim.mean().item()
    except Exception as e:
        logger.warning(f"Error computing cosine similarity: {e}")
        return 0.5  # Return middle value as fallback

# Add this to mlpdvae_utils.py (or wherever compute_embedding_similarity is defined)

def adapt_dimensions(tensor, target_dim):
    """Adapt tensor to target dimensions by padding or truncating"""
    # Ensure tensor is 2D
    if len(tensor.shape) == 1:
        tensor = tensor.unsqueeze(0)

    current_dim = tensor.shape[1]

    # Return unchanged if dimensions already match
    if current_dim == target_dim:
        return tensor

    # Handle dimension mismatch
    if current_dim < target_dim:
        # Pad with zeros
        padding = torch.zeros(tensor.shape[0], target_dim - current_dim, device=tensor.device)
        return torch.cat([tensor, padding], dim=1)
    else:
        # Truncate
        return tensor[:, :target_dim]
def generate_text_from_embedding(embedding, sentence_lookup=None):
    """
    A utility function to generate text from embeddings for semantic evaluation.

    Args:
        embedding: The embedding to convert to text
        sentence_lookup: Optional dictionary mapping embeddings to sentences

    Returns:
        Generated text or placeholder if not possible
    """
    # If we have a lookup dictionary, try to find the closest embedding
    if sentence_lookup is not None:
        # Convert embedding to tensor if needed
        if isinstance(embedding, np.ndarray):
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
        else:
            embedding_tensor = embedding

        # Find closest embedding in lookup
        best_match = None
        best_similarity = -1

        for emb, sentence in sentence_lookup.items():
            try:
                # Convert key to tensor if it's a tuple
                if isinstance(emb, tuple):
                    emb = torch.tensor(emb, dtype=torch.float32)

                similarity = compute_embedding_similarity(embedding_tensor, emb)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = sentence
            except Exception as e:
                logger.debug(f"Error comparing embeddings: {e}")
                continue

        if best_match is not None:
            return best_match

    # If we can't generate text, return a placeholder
    return "Generated text placeholder"


def detect_embedding_dimensions(data_dir="./data"):
    """
    Detect embedding dimensions from compressed data.

    Args:
        data_dir: Directory containing compressed data

    Returns:
        Tuple of (input_dim, hidden_dim, latent_dim)
    """
    import pickle
    import os

    compressed_data_path = os.path.join(data_dir, "compressed_data.pkl")

    try:
        with open(compressed_data_path, "rb") as f:
            compressed_data = pickle.load(f)

        if not compressed_data:
            logger.error("No compressed data found")
            return None, None, None

        # Get first item
        if isinstance(compressed_data[0], dict) and 'embedding' in compressed_data[0]:
            first_embedding = compressed_data[0]['embedding']
        elif isinstance(compressed_data[0], tuple) and len(compressed_data[0]) == 2:
            first_embedding = compressed_data[0][0]
        else:
            first_embedding = compressed_data[0]

        # Get dimensions
        input_dim = len(first_embedding)
        hidden_dim = min(1024, input_dim * 2)
        latent_dim = min(512, input_dim)

        logger.info(f"Detected dimensions: input={input_dim}, hidden={hidden_dim}, latent={latent_dim}")
        return input_dim, hidden_dim, latent_dim
    except Exception as e:
        logger.error(f"Error detecting dimensions: {e}")
        return None, None, None


def ensure_tensor_shape(tensor, expected_dim=2):
    """
    Ensure tensor has the correct shape.

    Args:
        tensor: Tensor or ndarray to check/reshape
        expected_dim: Expected number of dimensions (default 2 for [batch, features])

    Returns:
        Correctly shaped tensor
    """
    # Convert numpy to tensor if needed
    if isinstance(tensor, np.ndarray):
        tensor = torch.tensor(tensor, dtype=torch.float32)

    # Handle dimensionality
    current_dims = len(tensor.shape)

    # Add dimensions if needed
    if current_dims < expected_dim:
        for _ in range(expected_dim - current_dims):
            tensor = tensor.unsqueeze(0)

    # Remove dimensions if needed
    elif current_dims > expected_dim:
        # Only squeeze dimensions of size 1
        for dim in range(current_dims - expected_dim):
            if tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            else:
                # Can't reduce further without losing data
                break

    return tensor

def apply_noise_to_embedding(embedding, noise_level=0.05, noise_type='gaussian'):
    """
    Apply noise to embedding to simulate channel effects.

    Args:
        embedding: Embedding to corrupt
        noise_level: Level of noise to apply (0-1)
        noise_type: Type of noise ('gaussian', 'burst', 'dropout')

    Returns:
        Noisy embedding
    """
    import random

    # Convert to numpy if tensor
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.cpu().numpy()

    # Make a copy to avoid modifying the original
    noisy_embedding = embedding.copy()

    if noise_type == 'gaussian':
        # Add Gaussian noise scaled by embedding magnitude
        std_dev = np.std(embedding) * noise_level * 2
        noise = np.random.normal(0, std_dev, embedding.shape)
        noisy_embedding = embedding + noise

    elif noise_type == 'burst':
        # Burst noise affects a continuous segment of the embedding
        burst_length = int(len(embedding) * noise_level * 3)
        burst_start = random.randint(0, max(0, len(embedding) - burst_length - 1))
        burst_end = burst_start + burst_length

        # Higher intensity noise in burst region
        std_dev = np.std(embedding) * noise_level * 4
        noise = np.random.normal(0, std_dev, burst_length)
        noisy_embedding[burst_start:burst_end] += noise

    elif noise_type == 'dropout':
        # Randomly zero out elements (simulates packet loss)
        mask = np.random.random(embedding.shape) > noise_level
        noisy_embedding = embedding * mask

    return noisy_embedding