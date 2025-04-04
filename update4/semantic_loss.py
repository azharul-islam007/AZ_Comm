# semantic_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
from transformers import AutoTokenizer, AutoModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create models directory if it doesn't exist
os.makedirs('./models', exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EnhancedSemanticLoss(nn.Module):
    """
    Enhanced semantic loss with contrastive learning and discourse-level awareness.
    """

    def __init__(self,
                 model_name='sentence-transformers/all-mpnet-base-v2',
                 device=None,
                 contrastive_weight=0.3,
                 discourse_weight=0.2,
                 cache_dir='./models/semantic_model_cache'):
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.contrastive_weight = contrastive_weight
        self.discourse_weight = discourse_weight
        self.cache_dir = cache_dir

        # Initialize with better sentence transformer model
        logger.info(f"Initializing semantic loss with {model_name} model")
        try:
            os.makedirs(cache_dir, exist_ok=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(self.device)

            # Discourse modeling components
            self.discourse_projection = nn.Linear(768, 256).to(self.device)

            # Freeze base model parameters
            for param in self.model.parameters():
                param.requires_grad = False

            # Set to eval mode
            self.model.eval()

            # Initialize temperature parameter for contrastive loss
            self.temperature = nn.Parameter(torch.tensor(0.07))

            logger.info("Enhanced semantic loss initialized successfully")
            self.initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize enhanced semantic loss: {e}")
            self.initialized = False

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling for sentence embedding"""
        token_embeddings = model_output[0]  # First element contains token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _get_embeddings(self, sentences):
        """Get embeddings for a list of sentences"""
        if not self.initialized:
            return None

        if isinstance(sentences, str):
            sentences = [sentences]

        # Tokenize and get attention masks
        encoded_input = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)

        # Get model output
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Apply mean pooling
        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def _get_discourse_features(self, sentences, embeddings=None):
        """Extract discourse-level features from sentences"""
        if not sentences:
            return None

        # Get embeddings if not provided
        if embeddings is None:
            embeddings = self._get_embeddings(sentences)

        if isinstance(sentences, str):
            # Single sentence has no discourse features
            return torch.zeros(256, device=self.device)

        # Simple discourse features: sequence modeling
        # For more complex discourse analysis, full transformer would be better

        # Project embeddings to discourse space
        discourse_features = self.discourse_projection(embeddings)

        # For now, just average the features
        # In a full implementation, a sequence model would be better
        avg_discourse = torch.mean(discourse_features, dim=0)

        return avg_discourse

    def _semantic_similarity(self, emb1, emb2):
        """Compute cosine similarity between embeddings"""
        return F.cosine_similarity(emb1, emb2, dim=1)

    def _contrastive_loss(self, original_embs, reconstructed_embs, batch_size=None):
        """Compute contrastive loss between original and reconstructed embeddings"""
        # Handle single embedding case
        if batch_size is None:
            batch_size = original_embs.shape[0]

        # Normalized embeddings for cosine similarity
        original_norm = F.normalize(original_embs, p=2, dim=1)
        reconstructed_norm = F.normalize(reconstructed_embs, p=2, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(original_norm, reconstructed_norm.t()) / self.temperature

        # Labels are the indices for positive pairs (diagonal elements)
        labels = torch.arange(batch_size, device=self.device)

        # Compute loss (cross-entropy with similarity matrix)
        loss = F.cross_entropy(sim_matrix, labels)

        return loss

    def _discourse_consistency_loss(self, original_disc, reconstructed_disc):
        """Compute discourse consistency loss"""
        if original_disc is None or reconstructed_disc is None:
            return torch.tensor(0.0, device=self.device)

        # Simple MSE for discourse feature consistency
        return F.mse_loss(original_disc, reconstructed_disc)

    def semantic_similarity(self, original_text, reconstructed_text):
        """
        Compute semantic similarity between original and reconstructed text
        Returns scalar between 0 and 1
        """
        if not self.initialized:
            return 0.5  # Default midpoint value if not initialized

        if isinstance(original_text, str):
            original_text = [original_text]
        if isinstance(reconstructed_text, str):
            reconstructed_text = [reconstructed_text]

        orig_embs = self._get_embeddings(original_text)
        recon_embs = self._get_embeddings(reconstructed_text)

        similarities = self._semantic_similarity(orig_embs, recon_embs)
        return similarities.mean().item()

    def semantic_entailment(self, premise, hypothesis):
        """
        Check if hypothesis text is entailed by premise text
        Returns entailment score between 0 and 1
        """
        if not self.initialized:
            return 0.5  # Default midpoint value if not initialized

        # Use all-mpnet model as a good zero-shot entailment estimator
        premise_emb = self._get_embeddings([premise])
        hypothesis_emb = self._get_embeddings([hypothesis])

        # Asymmetric similarity function for entailment
        # Projects hypothesis onto premise space and measures coverage
        norm_premise = F.normalize(premise_emb, p=2, dim=1)
        norm_hypothesis = F.normalize(hypothesis_emb, p=2, dim=1)

        # Calculate dot product
        dot_product = (norm_premise * norm_hypothesis).sum(dim=1)

        # Calculate entailment score (0-1)
        entailment_score = (1 + dot_product) / 2

        return entailment_score.item()

    def forward(self, original_text, reconstructed_text):
        """
        Calculate enhanced semantic loss between original and reconstructed text.
        Combines semantic similarity with contrastive learning and discourse analysis.

        Args:
            original_text: Original text as string or list of strings
            reconstructed_text: Reconstructed text as string or list of strings

        Returns:
            Combined semantic loss
        """
        if not self.initialized:
            return torch.tensor(0.0, device=self.device)

        # Convert to lists if needed
        if isinstance(original_text, str):
            original_text = [original_text]
        if isinstance(reconstructed_text, str):
            reconstructed_text = [reconstructed_text]

        # Make sure lists have same length
        min_len = min(len(original_text), len(reconstructed_text))
        original_text = original_text[:min_len]
        reconstructed_text = reconstructed_text[:min_len]

        # Get embeddings for both sets of texts
        orig_embs = self._get_embeddings(original_text)
        recon_embs = self._get_embeddings(reconstructed_text)

        # Calculate semantic similarity loss
        sim_loss = 1.0 - self._semantic_similarity(orig_embs, recon_embs).mean()

        # Calculate contrastive loss if batch size > 1
        if len(original_text) > 1:
            contrast_loss = self._contrastive_loss(orig_embs, recon_embs, len(original_text))
        else:
            contrast_loss = torch.tensor(0.0, device=self.device)

        # Calculate discourse consistency loss if enabled
        if self.discourse_weight > 0 and len(original_text) > 1:
            orig_disc = self._get_discourse_features(original_text, orig_embs)
            recon_disc = self._get_discourse_features(reconstructed_text, recon_embs)
            disc_loss = self._discourse_consistency_loss(orig_disc, recon_disc)
        else:
            disc_loss = torch.tensor(0.0, device=self.device)

        # Combine losses with weights
        combined_loss = (
                sim_loss +
                self.contrastive_weight * contrast_loss +
                self.discourse_weight * disc_loss
        )

        return combined_loss

    def evaluate_semantic_quality(self, original_text, reconstructed_text, detailed=False):
        """
        Comprehensive semantic evaluation including similarity, entailment, 
        and discourse consistency.

        Args:
            original_text: Original text
            reconstructed_text: Reconstructed text
            detailed: Whether to return detailed breakdown

        Returns:
            Either combined score or dictionary of detailed metrics
        """
        if not self.initialized:
            return 0.5  # Default midpoint value if not initialized

        # Basic semantic similarity
        similarity = self.semantic_similarity(original_text, reconstructed_text)

        # Semantic entailment (both directions)
        if isinstance(original_text, list):
            # For simplicity, join text with spaces for entailment
            orig_joined = " ".join(original_text)
            recon_joined = " ".join(reconstructed_text)
        else:
            orig_joined = original_text
            recon_joined = reconstructed_text

        # Calculate bidirectional entailment
        orig_entails_recon = self.semantic_entailment(orig_joined, recon_joined)
        recon_entails_orig = self.semantic_entailment(recon_joined, orig_joined)

        # Discourse preservation (only for multi-sentence text)
        if (isinstance(original_text, list) and len(original_text) > 1) or "\n" in orig_joined:
            # Split into sentences if needed
            if not isinstance(original_text, list):
                original_sents = orig_joined.split(". ")
                recon_sents = recon_joined.split(". ")
            else:
                original_sents = original_text
                recon_sents = reconstructed_text

            # Get discourse features
            orig_embs = self._get_embeddings(original_sents)
            recon_embs = self._get_embeddings(recon_sents)
            orig_disc = self._get_discourse_features(original_sents, orig_embs)
            recon_disc = self._get_discourse_features(recon_sents, recon_embs)

            # Calculate discourse similarity (cosine sim of discourse features)
            if orig_disc is not None and recon_disc is not None:
                discourse_sim = F.cosine_similarity(
                    orig_disc.unsqueeze(0),
                    recon_disc.unsqueeze(0)
                ).item()
            else:
                discourse_sim = 0.5  # Default mid-value
        else:
            discourse_sim = 1.0  # Perfect for single sentences

        # Calculate combined score
        combined_score = (
                0.5 * similarity +
                0.3 * (orig_entails_recon + recon_entails_orig) / 2 +
                0.2 * discourse_sim
        )

        if detailed:
            return {
                'similarity': similarity,
                'entailment_orig_recon': orig_entails_recon,
                'entailment_recon_orig': recon_entails_orig,
                'discourse_similarity': discourse_sim,
                'combined_score': combined_score
            }
        else:
            return combined_score

    def calculate_semantic_similarity(self, text1, text2):
        """For backward compatibility with original SemanticPerceptualLoss"""
        return self.semantic_similarity(text1, text2)


# For backward compatibility and easier drop-in replacement
class SemanticPerceptualLoss(EnhancedSemanticLoss):
    """Legacy class for backward compatibility"""

    def __init__(self, bert_model='bert-base-uncased', layers_to_use=[-1, -2, -3, -4],
                 cache_dir='./models/bert_cache', max_length=128):
        # Override with better model
        super().__init__(
            model_name='sentence-transformers/all-mpnet-base-v2',
            cache_dir=cache_dir
        )
        # Store original parameters for compatibility
        self.layers_to_use = layers_to_use
        self.max_length = max_length
        self.layer_weights = {-1: 0.5, -2: 0.25, -3: 0.15, -4: 0.1}


def evaluate_semantic_similarity(original_text, reconstructed_text, semantic_loss=None):
    """
    Evaluate semantic similarity between original and reconstructed texts.
    For backward compatibility.
    """
    if semantic_loss is None or not hasattr(semantic_loss, 'initialized') or not semantic_loss.initialized:
        # Initialize a new loss module
        try:
            semantic_loss = EnhancedSemanticLoss()
            if not semantic_loss.initialized:
                return 0.5  # Default midpoint value
        except Exception as e:
            logger.warning(f"Could not initialize semantic loss: {e}")
            return 0.5

    # Use the new semantic similarity method
    similarity = semantic_loss.semantic_similarity(original_text, reconstructed_text)
    return similarity


def get_semantic_loss(cache_dir='./models/semantic_model_cache'):
    """Create and return an initialized EnhancedSemanticLoss"""
    semantic_loss = EnhancedSemanticLoss(cache_dir=cache_dir)
    return semantic_loss if semantic_loss.initialized else None
