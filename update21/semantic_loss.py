import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
from transformers import BertTokenizer, BertModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create models directory if it doesn't exist
os.makedirs('./models', exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SemanticPerceptualLoss(nn.Module):
    """
    Loss function that directly measures semantic similarity using BERT features.
    Used to enhance DVAE training with semantically-aware loss signals.
    """

    def __init__(self, bert_model='bert-base-uncased', layers_to_use=[-1, -2, -3, -4],
                 cache_dir='./models/bert_cache', max_length=128):
        super().__init__()

        # Store configuration
        self.layers_to_use = layers_to_use
        self.max_length = max_length

        # Initialize BERT tokenizer and model
        logger.info(f"Initializing semantic loss with {bert_model} model")
        try:
            os.makedirs(cache_dir, exist_ok=True)
            self.tokenizer = BertTokenizer.from_pretrained(bert_model, cache_dir=cache_dir)
            self.model = BertModel.from_pretrained(bert_model, cache_dir=cache_dir,
                                                   output_hidden_states=True)

            # Move model to device
            self.model = self.model.to(device)

            # Freeze BERT parameters - we don't want to fine-tune BERT during DVAE training
            for param in self.model.parameters():
                param.requires_grad = False

            # Layer weighting (deeper layers more important for semantics)
            self.layer_weights = {-1: 0.5, -2: 0.25, -3: 0.15, -4: 0.1}

            logger.info("Semantic perceptual loss initialized successfully")
            self.initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize semantic loss: {e}")
            self.initialized = False

    def get_embeddings(self, texts):
        """Get BERT embeddings for a list of texts"""
        if not self.initialized:
            return None

        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize texts
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True,
                                truncation=True, max_length=self.max_length).to(device)

        # Get BERT features
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get hidden states from selected layers
        hidden_states = outputs.hidden_states
        layer_embeddings = []

        for layer_idx in self.layers_to_use:
            # Get features from this layer
            layer_output = hidden_states[layer_idx]

            # Average over tokens (excluding special tokens)
            # Use attention mask to properly average only over real tokens
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            sum_embeddings = torch.sum(layer_output * attention_mask, dim=1)
            sum_mask = torch.sum(attention_mask, dim=1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)  # Avoid division by zero
            avg_embeddings = sum_embeddings / sum_mask

            # Normalize
            norm = avg_embeddings.norm(p=2, dim=1, keepdim=True)
            normalized_embeddings = avg_embeddings / norm.clamp(min=1e-9)

            layer_embeddings.append(normalized_embeddings)

        return layer_embeddings

    def forward(self, original_texts, reconstructed_texts):
        """Calculate semantic similarity loss between original and reconstructed texts"""
        if not self.initialized or not original_texts or not reconstructed_texts:
            # Return zero loss if not initialized or no texts provided
            return torch.tensor(0.0, device=device)

        # Handle single string input
        if isinstance(original_texts, str):
            original_texts = [original_texts]
        if isinstance(reconstructed_texts, str):
            reconstructed_texts = [reconstructed_texts]

        # Get embeddings for both sets of texts
        orig_embeddings = self.get_embeddings(original_texts)
        recon_embeddings = self.get_embeddings(reconstructed_texts)

        if not orig_embeddings or not recon_embeddings:
            return torch.tensor(0.0, device=device)

        # Calculate loss across layers
        total_loss = 0.0
        for i, layer in enumerate(self.layers_to_use):
            # Get embeddings for this layer
            orig_layer_emb = orig_embeddings[i]
            recon_layer_emb = recon_embeddings[i]

            # Cosine similarity loss (1 - similarity)
            similarity = F.cosine_similarity(orig_layer_emb, recon_layer_emb)
            layer_loss = 1.0 - similarity.mean()

            # Weight by layer importance
            weight = self.layer_weights.get(layer, 0.25)
            total_loss += weight * layer_loss

        return total_loss

    def calculate_semantic_similarity(self, text1, text2):
        """
        Calculate semantic similarity between two texts.
        Returns a value between 0 and 1, where 1 means identical meaning.
        """
        if not self.initialized:
            return 0.5  # Default midpoint value if not initialized

        # Convert to list if necessary
        if isinstance(text1, str):
            text1 = [text1]
        if isinstance(text2, str):
            text2 = [text2]

        # Get embeddings
        text1_embeddings = self.get_embeddings(text1)
        text2_embeddings = self.get_embeddings(text2)

        if not text1_embeddings or not text2_embeddings:
            return 0.5

        # Calculate weighted similarity across layers
        similarity = 0.0
        total_weight = sum(self.layer_weights.values())

        for i, layer in enumerate(self.layers_to_use):
            # Get embeddings for this layer
            text1_emb = text1_embeddings[i]
            text2_emb = text2_embeddings[i]

            # Cosine similarity
            layer_sim = F.cosine_similarity(text1_emb, text2_emb).mean().item()

            # Weight by layer importance
            weight = self.layer_weights.get(layer, 0.25)
            similarity += (weight / total_weight) * layer_sim

        return similarity


def evaluate_semantic_similarity(original_text, reconstructed_text, semantic_loss=None):
    """
    Evaluate semantic similarity between original and reconstructed texts.

    Args:
        original_text: Original text
        reconstructed_text: Reconstructed text
        semantic_loss: Optional SemanticPerceptualLoss instance

    Returns:
        Semantic similarity score (0-1)
    """
    if semantic_loss is None or not hasattr(semantic_loss, 'initialized') or not semantic_loss.initialized:
        # Initialize a new loss module
        try:
            semantic_loss = SemanticPerceptualLoss()
            if not semantic_loss.initialized:
                return 0.5  # Default midpoint value
        except Exception as e:
            logger.warning(f"Could not initialize semantic loss: {e}")
            return 0.5

    similarity = semantic_loss.calculate_semantic_similarity(
        original_text, reconstructed_text)

    return similarity


def get_semantic_loss(cache_dir='./models/bert_cache'):
    """Create and return an initialized SemanticPerceptualLoss"""
    semantic_loss = SemanticPerceptualLoss(cache_dir=cache_dir)
    return semantic_loss if semantic_loss.initialized else None
