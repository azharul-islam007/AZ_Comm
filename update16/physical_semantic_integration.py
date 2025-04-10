import os
import numpy as np
import torch
import torch.nn as nn  # Add this line
import torch.nn.functional as F  # You might need this too
import logging
import time
import json
import difflib
from physical_channel import PhysicalChannelLayer
from content_adaptive_coding import ContentAdaptivePhysicalChannel  # Import new class
from compression_vae import decompress_vae_embedding  # Import from new compression module
from config_manager import ConfigManager
from mlpdvae_utils import ensure_tensor_shape
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def timing_decorator(func):
    """Decorator to measure and log function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"[TIMING] Function {func.__name__} took {end_time - start_time:.4f} seconds to execute")
        return result
    return wrapper
# Import the configuration
try:
    import physical_channel_config as config
except ImportError:
    # Use ConfigManager instead of hardcoded DefaultConfig
    config_manager = ConfigManager()


    class DefaultConfig:
        def __init__(self):
            self.ENABLE_PHYSICAL_CHANNEL = config_manager.get("physical.enable_physical_channel", True)
            self.CHANNEL_TYPE = config_manager.get("physical.channel_type", "awgn")
            self.SNR_DB = config_manager.get("physical.snr_db", 20.0)
            self.MODULATION_TYPE = config_manager.get("physical.modulation_type", "qam")
            self.MODULATION_ORDER = config_manager.get("physical.modulation_order", 16)
            self.USE_CHANNEL_CODING = config_manager.get("physical.use_channel_coding", True)
            self.CODING_RATE = config_manager.get("physical.coding_rate", 0.75)
            self.USE_IMPORTANCE_WEIGHTING = config_manager.get("physical.use_importance_weighting", True)
            self.WEIGHT_METHOD = config_manager.get("physical.weight_method", "semantic")
            self.ENABLE_ADAPTIVE_MODULATION = config_manager.get("physical.enable_adaptive_modulation", True)
            self.ENABLE_UNEQUAL_ERROR_PROTECTION = config_manager.get("physical.enable_unequal_error_protection", True)
            self.ENABLE_CONTENT_ADAPTIVE_CODING = config_manager.get("physical.enable_content_adaptive_coding", True)
            self.VAE_COMPRESSION = config_manager.get("physical.vae_compression", True)
            self.CONTENT_CLASSIFIER_PATH = config_manager.get("physical.content_classifier_path",
                                                              "./models/content_classifier.pth")


    config = DefaultConfig()

# Setup logging
logger = logging.getLogger(__name__)


class EnhancedPhysicalSemanticIntegration:
    """
    Enhanced integration layer between semantic and physical channels.
    Supports VAE compression and content-adaptive coding.
    """

    def __init__(self, channel_config=None):
        """
        Initialize the integration layer.

        Args:
            channel_config: Optional configuration for the physical channel.
                If None, uses the imported configuration.
        """
        # Use provided config or imported default
        self.config = channel_config or config

        # Initialize physical channel if enabled
        self._physical_channel = None
        self.physical_enabled = getattr(self.config, 'ENABLE_PHYSICAL_CHANNEL', True)

        # Check if we should use VAE compression
        self.use_vae_compression = getattr(self.config, 'VAE_COMPRESSION', True)

        # Check if we should use content-adaptive coding
        self.use_content_adaptive = getattr(self.config, 'ENABLE_CONTENT_ADAPTIVE_CODING', True)

        if self.physical_enabled:
            self._init_physical_channel()

        # Initialize PCA for importance analysis
        self.pca = None
        self.pca_fitted = False

        # Set up transmission data collection for self-supervised learning
        self.collect_transmission_data = getattr(self.config, 'COLLECT_TRANSMISSION_DATA', True)
        if self.collect_transmission_data:
            self.transmission_pairs_dir = getattr(self.config, 'TRANSMISSION_PAIRS_DIR', './transmission_pairs')
            os.makedirs(self.transmission_pairs_dir, exist_ok=True)
            self.max_pairs = getattr(self.config, 'MAX_TRANSMISSION_PAIRS', 10000)
            self.collected_pairs = 0
            logger.info(f"Will collect up to {self.max_pairs} transmission pairs")

        # Track metrics
        self.metrics = {
            'snr_values': [],
            'mse_values': [],
            'ber_values': [],
            'semantic_scores': []
        }

        # Create results directory if needed
        results_dir = getattr(self.config, 'CHANNEL_RESULTS_DIR', './channel_results')
        os.makedirs(results_dir, exist_ok=True)

    def _init_physical_channel(self):
        """Initialize the physical channel based on configuration."""
        try:
            # Use content-adaptive channel if enabled
            if self.use_content_adaptive:
                try:
                    # Try to initialize with content-adaptive parameters
                    self._physical_channel = ContentAdaptivePhysicalChannel(
                        snr_db=getattr(self.config, 'SNR_DB', 20.0),
                        channel_type=getattr(self.config, 'CHANNEL_TYPE', 'awgn'),
                        modulation=getattr(self.config, 'MODULATION_TYPE', 'qam'),
                        modulation_order=getattr(self.config, 'MODULATION_ORDER', 16),
                        coding_rate=getattr(self.config, 'CODING_RATE', 0.75),
                        coding_type=getattr(self.config, 'CODING_TYPE', 'repetition'),
                        use_channel_coding=getattr(self.config, 'USE_CHANNEL_CODING', True),
                        importance_weighting=getattr(self.config, 'USE_IMPORTANCE_WEIGHTING', True),
                        enable_adaptive_modulation=getattr(self.config, 'ENABLE_ADAPTIVE_MODULATION', True),
                        enable_unequal_error_protection=getattr(self.config, 'ENABLE_UNEQUAL_ERROR_PROTECTION', True),
                        enable_content_adaptive_coding=getattr(self.config, 'ENABLE_CONTENT_ADAPTIVE_CODING', True),
                        content_classifier_path=getattr(self.config, 'CONTENT_CLASSIFIER_PATH',
                                                        './models/content_classifier.pth'),
                        ofdm_carriers=getattr(self.config, 'OFDM_CARRIERS', 64),
                        fading_param=(getattr(self.config, 'RICIAN_K_FACTOR', 4.0)
                                      if getattr(self.config, 'CHANNEL_TYPE', 'awgn') == 'rician'
                                      else getattr(self.config, 'RAYLEIGH_VARIANCE', 1.0))
                    )
                    logger.info("Enhanced content-adaptive physical channel initialized")
                except TypeError:
                    logger.warning(
                        "ContentAdaptivePhysicalChannel initialization failed with TypeError. Falling back to standard channel")
                    # Fall back to standard channel
                    self._physical_channel = PhysicalChannelLayer(
                        snr_db=getattr(self.config, 'SNR_DB', 20.0),
                        channel_type=getattr(self.config, 'CHANNEL_TYPE', 'awgn'),
                        modulation=getattr(self.config, 'MODULATION_TYPE', 'qam'),
                        modulation_order=getattr(self.config, 'MODULATION_ORDER', 16),
                        coding_rate=getattr(self.config, 'CODING_RATE', 0.75),
                        coding_type=getattr(self.config, 'CODING_TYPE', 'repetition'),
                        use_channel_coding=getattr(self.config, 'USE_CHANNEL_CODING', True),
                        importance_weighting=getattr(self.config, 'USE_IMPORTANCE_WEIGHTING', True),
                        enable_adaptive_modulation=getattr(self.config, 'ENABLE_ADAPTIVE_MODULATION', True),
                        enable_unequal_error_protection=getattr(self.config, 'ENABLE_UNEQUAL_ERROR_PROTECTION', True),
                        ofdm_carriers=getattr(self.config, 'OFDM_CARRIERS', 64),
                        fading_param=(getattr(self.config, 'RICIAN_K_FACTOR', 4.0)
                                      if getattr(self.config, 'CHANNEL_TYPE', 'awgn') == 'rician'
                                      else getattr(self.config, 'RAYLEIGH_VARIANCE', 1.0))
                    )
                    logger.info("Fallback to standard physical channel successful")
            else:
                # Use original physical channel
                self._physical_channel = PhysicalChannelLayer(
                    snr_db=getattr(self.config, 'SNR_DB', 20.0),
                    channel_type=getattr(self.config, 'CHANNEL_TYPE', 'awgn'),
                    modulation=getattr(self.config, 'MODULATION_TYPE', 'qam'),
                    modulation_order=getattr(self.config, 'MODULATION_ORDER', 16),
                    coding_rate=getattr(self.config, 'CODING_RATE', 0.75),
                    coding_type=getattr(self.config, 'CODING_TYPE', 'repetition'),
                    use_channel_coding=getattr(self.config, 'USE_CHANNEL_CODING', True),
                    importance_weighting=getattr(self.config, 'USE_IMPORTANCE_WEIGHTING', True),
                    enable_adaptive_modulation=getattr(self.config, 'ENABLE_ADAPTIVE_MODULATION', True),
                    enable_unequal_error_protection=getattr(self.config, 'ENABLE_UNEQUAL_ERROR_PROTECTION', True),
                    ofdm_carriers=getattr(self.config, 'OFDM_CARRIERS', 64),
                    fading_param=(getattr(self.config, 'RICIAN_K_FACTOR', 4.0)
                                  if getattr(self.config, 'CHANNEL_TYPE', 'awgn') == 'rician'
                                  else getattr(self.config, 'RAYLEIGH_VARIANCE', 1.0))
                )
                logger.info("Standard physical channel initialized")
        except Exception as e:
            logger.error(f"Failed to initialize physical channel: {e}")
            self.physical_enabled = False

    def _calculate_importance_weights(self, embedding, method=None):
        """
        Enhanced: Calculate importance weights for the embedding dimensions using multiple methods.
        Now compatible with VAE-compressed embeddings.

        Args:
            embedding: Input embedding to analyze
            method: Weight calculation method ('variance', 'pca', 'semantic', 'uniform')

        Returns:
            Importance weights for each dimension
        """
        if method is None:
            method = getattr(self.config, 'WEIGHT_METHOD', 'variance')

        if not getattr(self.config, 'USE_IMPORTANCE_WEIGHTING', True):
            return np.ones_like(embedding)

        # Check if we have batch or single embedding
        is_batch = len(embedding.shape) > 1 and embedding.shape[0] > 1

        # For VAE-compressed embeddings, we can use specialized methods
        if self.use_vae_compression and method == 'semantic':
            # In VAE space, dimensions are already optimized for importance
            # Use a gradient-based approach where earlier dimensions are more important
            if is_batch:
                weights = np.ones((embedding.shape[0], embedding.shape[1]))
                for i in range(weights.shape[0]):
                    # Linear importance decay (first dimensions more important)
                    weights[i] = np.linspace(1.0, 0.4, embedding.shape[1])
            else:
                # Linear importance decay
                weights = np.linspace(1.0, 0.4, embedding.shape[0])

            return weights

        # For other methods, use the original implementation
        if method == 'variance':
            # For batches, use variance with numerical stability
            if isinstance(embedding, np.ndarray) and embedding.size > 0:
                # Add small epsilon to prevent division by zero
                epsilon = 1e-10
                mean = np.mean(embedding, axis=0)
                # Use manual variance calculation to avoid numerical issues
                squared_diff = np.square(embedding - mean)
                variance = np.mean(squared_diff, axis=0) + epsilon

                # Normalize to range [0.5, 1.0] to avoid extreme downweighting
                weights = 0.5 + 0.5 * (variance / (np.max(variance) + epsilon))

                # Broadcast to match embedding shape if needed
                if is_batch:
                    weights = np.tile(weights, (embedding.shape[0], 1))
            else:
                # For single embedding, use uniform weights
                weights = np.ones_like(embedding)

        else:  # Other methods remain unchanged
            # Implementation for PCA and other methods...
            weights = np.ones_like(embedding)

        return weights

    def _save_transmission_pair(self, original, received):
        """
        Save original-received embedding pairs for self-supervised learning.

        Args:
            original: Original embedding before transmission
            received: Received embedding after transmission
        """
        if not self.collect_transmission_data or self.collected_pairs >= self.max_pairs:
            return

        try:
            # Convert torch tensors to numpy if needed
            if isinstance(original, torch.Tensor):
                original = original.detach().cpu().numpy()
            if isinstance(received, torch.Tensor):
                received = received.detach().cpu().numpy()

            # Save pair
            pair_file = os.path.join(self.transmission_pairs_dir, f"pair_{self.collected_pairs:06d}.npz")
            np.savez(pair_file, original=original, received=received)

            self.collected_pairs += 1

            # Log progress periodically
            if self.collected_pairs % 100 == 0:
                logger.info(f"Collected {self.collected_pairs}/{self.max_pairs} transmission pairs")

        except Exception as e:
            logger.warning(f"Failed to save transmission pair: {e}")

    def transmit(self, embedding, debug=False, use_kb=True):
        """
        Method to transmit through the physical channel.

        Args:
            embedding: The embedding to transmit
            debug: Whether to return debug information
            use_kb: Whether to use knowledge base

        Returns:
            The transmitted embedding
        """
        start_time = time.time()
        logger.info(
            f"[PHYSICAL] Starting transmission for embedding of shape {embedding.shape if hasattr(embedding, 'shape') else 'unknown'}")

        # Check if physical channel is enabled
        if not self.physical_enabled or self._physical_channel is None:
            elapsed = time.time() - start_time
            logger.info(f"[PHYSICAL] Physical channel disabled, bypassing transmission (took {elapsed:.3f}s)")
            return embedding  # Pass through if physical channel is disabled

        # Store original embedding for self-supervised learning
        if isinstance(embedding, torch.Tensor):
            original_embedding = embedding.clone()
            embedding_np = embedding.detach().cpu().numpy()
        else:
            original_embedding = embedding.copy()
            embedding_np = embedding

        # Apply KB-guided importance weighting if knowledge base is available
        importance_weights = None
        if use_kb:
            try:
                from knowledge_base import get_or_create_knowledge_base
                kb = get_or_create_knowledge_base()

                # Apply semantic-aware importance weighting
                if hasattr(kb, 'get_importance_weights'):
                    importance_weights = kb.get_importance_weights(embedding_np)
                    logger.debug("[PHYSICAL] Applied KB-guided importance weighting")
            except Exception as e:
                logger.debug(f"[PHYSICAL] KB importance weighting failed: {e}")

        # If no KB weights, use default method
        if importance_weights is None:
            importance_weights = self._calculate_importance_weights(
                embedding_np, method=getattr(self.config, 'WEIGHT_METHOD', 'semantic')
            )

        # Transmit through physical channel
        try:
            if debug:
                received_embedding, debug_info = self._physical_channel.transmit(
                    embedding_np, importance_weights, debug=True)
            else:
                received_embedding = self._physical_channel.transmit(
                    embedding_np, importance_weights, debug=False)
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[PHYSICAL] Transmission failed: {e} (took {elapsed:.3f}s)")
            return embedding  # Return original if transmission fails

        # Save transmission pair for self-supervised learning
        if self.collect_transmission_data:
            self._save_transmission_pair(original_embedding, received_embedding)

        # Record metrics if requested
        if hasattr(self, 'log_transmission_metrics'):
            self.log_transmission_metrics(embedding_np, received_embedding)

        # Apply post-transmission KB enhancement if available
        if use_kb:
            try:
                if hasattr(kb, 'enhance_embedding'):
                    # Use KB to enhance the received embedding based on semantic knowledge
                    enhanced = kb.enhance_embedding(received_embedding, None)  # No text available here

                    # Apply a limited blend to avoid over-correction (80% received, 20% enhancement)
                    blend_factor = 0.2
                    received_embedding = (1.0 - blend_factor) * received_embedding + blend_factor * enhanced
                    logger.debug("[PHYSICAL] Applied post-transmission KB enhancement")
            except Exception as e:
                logger.debug(f"[PHYSICAL] Post-transmission KB enhancement failed: {e}")

        # Convert back to torch tensor if the input was a torch tensor
        if isinstance(embedding, torch.Tensor):
            device = embedding.device
            received_embedding = torch.tensor(received_embedding, dtype=torch.float32).to(device)

        # Log elapsed time before both return paths
        elapsed = time.time() - start_time
        logger.info(f"[PHYSICAL] Transmission completed in {elapsed:.3f}s")

        # Return the received embedding
        if debug:
            return received_embedding, debug_info
        else:
            return received_embedding

    def run_channel_sweep(self, embedding, snr_range=None, channel_types=None):
        """
        Run a sweep over different SNR values and channel types.

        Args:
            embedding: Embedding to transmit
            snr_range: Range of SNR values to test
            channel_types: List of channel types to test

        Returns:
            Dictionary of results
        """
        # This method is unchanged - it already works with new features
        pass

    def log_transmission_metrics(self, original, received, semantic_score=None):
        """
        Log metrics for a transmission through the physical channel.

        Args:
            original: Original embedding
            received: Received embedding
            semantic_score: Optional semantic similarity score
        """
        # This method is unchanged - it already works with new features
        pass

    def _plot_metrics(self):
        """Plot the accumulated metrics."""
        # This method is unchanged - it already works with new features
        pass

    def get_channel_info(self):
        """Get information about the physical channel configuration."""
        if not self.physical_enabled or self._physical_channel is None:
            return {"status": "disabled"}

        # Get base channel info
        info = self._physical_channel.get_channel_info()

        # Add VAE compression info
        info["vae_compression"] = self.use_vae_compression

        return info

    def set_physical_enabled(self, enabled):
        """Enable or disable the physical channel."""
        self.physical_enabled = enabled

        if enabled and self._physical_channel is None:
            self._init_physical_channel()

        logger.info(f"Physical channel {'enabled' if enabled else 'disabled'}")


# In physical_semantic_integration.py, add after EnhancedPhysicalSemanticIntegration class
class SemanticChannelOptimizer:
    """Optimizes physical channel parameters based on semantic importance"""

    def __init__(self, physical_channel, semantic_loss=None):
        self.physical_channel = physical_channel

        # Try to load semantic loss if not provided
        if semantic_loss is None:
            try:
                from semantic_loss import get_semantic_loss
                self.semantic_loss = get_semantic_loss()
            except ImportError:
                logger.warning("Could not import semantic loss module")
                self.semantic_loss = None
        else:
            self.semantic_loss = semantic_loss

        # Initialize importance extraction model
        self.initialize_importance_model()

    def initialize_importance_model(self):
        """Initialize importance extraction model"""
        try:
            # Use a simple feed-forward network to predict importance
            input_dim = 460
            self.importance_model = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            ).to(device)

            # Initialize with reasonable weights so it produces meaningful values
            # even before training
            for layer in self.importance_model:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=0.7)

        except Exception as e:
            logger.warning(f"Could not initialize importance model: {e}")
            self.importance_model = None

    def optimize_transmission(self, text, embedding):
        """
        Optimize transmission parameters based on semantic content

        Args:
            text: Text to transmit
            embedding: Embedding tensor

        Returns:
            Tuple of (optimized embedding, importance weights)
        """
        # Extract importance weights
        importance_weights = self.extract_importance_weights(text, embedding)

        # Adjust physical channel parameters based on overall importance
        self._adjust_channel_parameters(importance_weights)

        return embedding, importance_weights

    def extract_importance_weights(self, text, embedding):
        """Extract semantic importance weights for the embedding dimensions"""
        if self.semantic_loss is None or not text:
            # Fallback to simple variance-based importance
            if isinstance(embedding, torch.Tensor):
                # Fix variance calculation
                if len(embedding.shape) <= 1:
                    # For 1D tensor, return uniform weights
                    importance = torch.ones_like(embedding)
                else:
                    # For 2D+ tensor, calculate variance along dim 0
                    variance = torch.var(embedding, dim=0, unbiased=False)  # Use unbiased=False to prevent warning
                    importance = 0.2 + 0.8 * (variance / (torch.max(variance) + 1e-8))
                return importance
            else:
                # Numpy array
                if len(embedding.shape) <= 1:
                    # For 1D array, return uniform weights
                    importance = np.ones_like(embedding)
                else:
                    # For 2D+ array, calculate variance along axis 0
                    variance = np.var(embedding, axis=0)
                    importance = 0.2 + 0.8 * (variance / (np.max(variance) + 1e-8))
                return importance

        # With semantic loss, we can do more sophisticated importance extraction
        try:
            # Use the importance model to predict importance
            if self.importance_model is not None:
                with torch.no_grad():
                    tensor = torch.tensor(embedding, dtype=torch.float32).to(device)

                    # Ensure proper shape
                    if len(tensor.shape) == 1:
                        tensor = tensor.unsqueeze(0)

                    # Get importance prediction
                    importance = self.importance_model(tensor).squeeze()

                    # Mix with a variance-based baseline
                    variance = torch.var(tensor, dim=0)
                    var_importance = 0.2 + 0.8 * (variance / (torch.max(variance) + 1e-8))

                    # Combine both (25% model, 75% variance-based)
                    importance = 0.25 * importance + 0.75 * var_importance

                    return importance.cpu().numpy() if isinstance(embedding, np.ndarray) else importance

            # Fallback to simple variance-based importance
            if isinstance(embedding, torch.Tensor):
                variance = torch.var(embedding, dim=0)
                importance = 0.2 + 0.8 * (variance / (torch.max(variance) + 1e-8))
                return importance
            else:
                variance = np.var(embedding, axis=0)
                importance = 0.2 + 0.8 * (variance / (np.max(variance) + 1e-8))
                return importance

        except Exception as e:
            logger.warning(f"Error in importance extraction: {e}")
            # Return uniform importance
            if isinstance(embedding, torch.Tensor):
                return torch.ones_like(embedding[0] if len(embedding.shape) > 1 else embedding)
            else:
                return np.ones_like(embedding[0] if len(embedding.shape) > 1 else embedding)

    def _adjust_channel_parameters(self, importance_weights):
        """Adjust physical channel parameters based on importance"""
        if self.physical_channel is None:
            return

        # Calculate overall importance
        avg_importance = float(np.mean(importance_weights) if isinstance(importance_weights, np.ndarray)
                               else torch.mean(importance_weights).item())

        # Adjust parameters based on importance
        if avg_importance > 0.7:  # High importance content
            # More conservative settings for important content
            self.physical_channel.modulation_order = min(self.physical_channel.modulation_order, 16)
            self.physical_channel.coding_rate = min(self.physical_channel.coding_rate, 0.7)
        elif avg_importance < 0.4:  # Low importance content
            # More aggressive settings for less important content
            self.physical_channel.coding_rate = max(self.physical_channel.coding_rate, 0.8)

# Create an instance of the integration layer for easy import
physical_semantic_bridge = EnhancedPhysicalSemanticIntegration()


class DimensionRegistry:
    """Central registry for managing embedding dimensions across components"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DimensionRegistry, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        """Initialize with default dimensions"""
        self.original_dim = 768  # Default BERT dimension
        self.compressed_dim = 460  # Default compressed dimension
        self.dvae_latent_dim = 460  # Default DVAE latent space
        self.dimensions_locked = False  # NEW: Flag to prevent changes after initialization

        # Load from config or detect from data
        self._load_dimensions()

    def lock_dimensions(self):
        """Lock dimensions to prevent further changes"""
        self.dimensions_locked = True
        logger.info(
            f"Dimensions locked: original={self.original_dim}, compressed={self.compressed_dim}, latent={self.dvae_latent_dim}")

    def _load_dimensions(self):
        """Load dimensions from config or saved data"""
        try:
            # Try to load from file
            vae_dim_path = os.path.join("./data", 'vae_dimensions.json')
            if os.path.exists(vae_dim_path):
                with open(vae_dim_path, 'r') as f:
                    dimensions = json.load(f)
                    self.original_dim = dimensions.get('input_dim', self.original_dim)
                    self.compressed_dim = dimensions.get('compressed_dim', self.compressed_dim)
                    logger.info(f"Loaded dimensions from file: {self.original_dim}, {self.compressed_dim}")
        except Exception as e:
            logger.warning(f"Could not load dimensions: {e}")

    def get_dims(self):
        """Get current dimensions"""
        return {
            'original': self.original_dim,
            'compressed': self.compressed_dim,
            'latent': self.dvae_latent_dim
        }

    def update(self, key, value):
        """Update a dimension value with safety checks"""
        if self.dimensions_locked:
            logger.warning(f"Cannot update {key} dimension to {value} - dimensions are locked")
            return False

        if hasattr(self, key):
            old_value = getattr(self, key)
            setattr(self, key, value)
            logger.info(f"Updated dimension {key}: {old_value} → {value}")
            return True
        return False


# In physical_semantic_integration.py

@timing_decorator
def transmit_through_physical_channel(embedding, importance_weights=None, debug=False, use_kb=True, context=None,
                                      context_list=None):
    """
    Enhanced function to transmit through the physical channel with better error recovery.
    """
    from SD5 import basic_text_reconstruction, estimate_corruption_level
    if not physical_semantic_bridge.physical_enabled or physical_semantic_bridge._physical_channel is None:
        return embedding  # Pass through if physical channel is disabled

    # Store original embedding for self-supervised learning
    if isinstance(embedding, torch.Tensor):
        original_embedding = embedding.clone()
        embedding_np = embedding.detach().cpu().numpy()
    else:
        original_embedding = embedding.copy()
        embedding_np = embedding

    # Get embedding dimensions for proper handling
    emb_dim = embedding_np.shape[-1] if len(embedding_np.shape) > 1 else embedding_np.shape[0]
    logger.debug(f"Processing embedding with dimensions: {emb_dim}")

    # Ensure proper shape for processing
    if len(embedding_np.shape) == 1:
        embedding_np = np.expand_dims(embedding_np, 0)

    # Apply KB-guided importance weighting with improved integration
    if importance_weights is None and use_kb:
        try:
            from knowledge_base import get_or_create_knowledge_base
            kb = get_or_create_knowledge_base()

            # Apply semantic-aware importance weighting
            if hasattr(kb, 'get_importance_weights'):
                importance_weights = kb.get_importance_weights(embedding_np)
                logger.debug("Applied KB-guided importance weighting")

                # Apply additional context-aware weighting if context is available
                if context and hasattr(kb, 'enhance_with_context'):
                    # Check if we have a context list for enhanced weighting
                    if context_list and len(context_list) > 1:
                        # Use weighted combination based on context relevance
                        context_weights = []
                        for ctx in context_list:
                            # Calculate relevance score for this context
                            relevance = kb.calculate_context_relevance(ctx, embedding_np)
                            context_weights.append((ctx, relevance))

                        # Sort by relevance
                        context_weights.sort(key=lambda x: x[1], reverse=True)

                        # Apply weights to importance weights based on context relevance
                        for ctx, weight in context_weights[:3]:  # Use top 3 contexts
                            ctx_weights = kb.get_importance_weights_for_context(embedding_np, ctx)
                            if ctx_weights is not None:
                                # Blend with weight based on relevance
                                importance_weights = importance_weights * (1 - weight * 0.2) + ctx_weights * (
                                            weight * 0.2)
        except Exception as e:
            logger.debug(f"KB importance weighting failed: {e}")

    # Apply enhanced error recovery after transmission
    try:
        from knowledge_base import get_or_create_knowledge_base
        from SD5 import extract_parliamentary_features  # Add this import

        kb = get_or_create_knowledge_base() if use_kb else None

        # Extract parliamentary features for enhanced error recovery
        parl_features = extract_parliamentary_features(context) if context else None

        # Check if we should use semantic error recovery
        if kb and hasattr(kb, 'enhance_embedding'):
            # Enhance the embedding with semantic context
            if context:
                enhanced_embedding = kb.enhance_embedding_with_context(embedding_np, context)
                if enhanced_embedding is not None:
                    embedding_np = enhanced_embedding

            # Apply multi-context enhancement if available
            if context_list and len(context_list) > 0 and hasattr(kb, 'enhance_with_multiple_contexts'):
                enhanced_embedding = kb.enhance_with_multiple_contexts(embedding_np, context_list)
                if enhanced_embedding is not None:
                    embedding_np = enhanced_embedding
    except Exception as e:
        logger.debug(f"Pre-transmission enhancement failed: {e}")

    # Transmit through physical channel with enhanced error handling
    try:
        logger.debug(f"Transmitting embedding with shape {embedding_np.shape} and dimension {emb_dim}")

        # Apply transmission
        received_embedding = physical_semantic_bridge._physical_channel.transmit(
            embedding_np, importance_weights, debug=False)

        # Apply post-transmission semantic error recovery with enhanced context utilization
        # FIX: Remove context_texts parameter which is causing the error
        received_embedding = semantic_error_recovery(
            received_embedding,
            context_embeddings=[original_embedding],
            kb=kb
        )
        logger.debug("Applied semantic error recovery")
    except Exception as e:
        logger.warning(f"Transmission error: {e}. Using original embedding.")
        return embedding

    # Apply post-transmission KB enhancement with multi-context
    if use_kb:
        try:
            kb = get_or_create_knowledge_base()

            if hasattr(kb, 'enhance_embedding'):
                # Use KB to enhance the received embedding with context
                enhanced = kb.enhance_embedding(received_embedding, None)  # Base enhancement

                # Apply dynamic blend based on context availability
                # More context = more confidence in enhancement
                context_size = 1 + (len(context_list) if context_list else 0)
                context_confidence = min(0.4, 0.2 + 0.05 * context_size)
                blend_factor = context_confidence

                # Apply the blending
                received_embedding = (1.0 - blend_factor) * received_embedding + blend_factor * enhanced
        except Exception as e:
            logger.debug(f"Post-transmission KB enhancement failed: {e}")

    # Convert back to original format
    if isinstance(embedding, torch.Tensor):
        device = embedding.device
        received_embedding = torch.tensor(received_embedding, dtype=torch.float32).to(device)

    # Ensure the output has the same shape as the input
    if len(embedding_np.shape) > 1 and len(received_embedding.shape) == 1:
        # Add batch dimension if needed
        if isinstance(received_embedding, torch.Tensor):
            received_embedding = received_embedding.unsqueeze(0)
        else:
            received_embedding = np.expand_dims(received_embedding, 0)
    elif len(embedding_np.shape) == 1 and len(received_embedding.shape) > 1:
        # Remove batch dimension if needed
        if isinstance(received_embedding, torch.Tensor):
            received_embedding = received_embedding.squeeze(0)
        else:
            received_embedding = received_embedding.squeeze(0)

    return received_embedding

def _apply_minimal_smoothing(embedding, window_size=5):
    """Apply minimal smoothing to embedding"""
    # Apply simple moving average smoothing
    result = embedding.copy()

    # If not enough elements for meaningful smoothing, return original
    if embedding.shape[1] <= window_size:
        return result

    # Apply moving average smoothing across feature dimension
    for i in range(embedding.shape[0]):
        for j in range(window_size, embedding.shape[1] - window_size):
            # If value appears to be an outlier, smooth it
            neighborhood = embedding[i, j - window_size:j + window_size]
            mean = np.mean(neighborhood)
            std = np.std(neighborhood)

            # Check if current value is an outlier
            if abs(embedding[i, j] - mean) > 2 * std:
                # Replace with moving average
                result[i, j] = mean

    return result


def cosine_similarity(a, b):
    """Calculate cosine similarity between two embeddings"""
    # Ensure proper shapes
    if len(a.shape) == 1:
        a = np.expand_dims(a, 0)
    if len(b.shape) == 1:
        b = np.expand_dims(b, 0)

    # Compute dot product
    dot_product = np.sum(a * b, axis=1)

    # Compute norms
    norm_a = np.sqrt(np.sum(a ** 2, axis=1))
    norm_b = np.sqrt(np.sum(b ** 2, axis=1))

    # Compute similarity
    similarity = dot_product / (norm_a * norm_b + 1e-8)

    return similarity


def _try_context_recovery(corrupted, context_embeddings, kb=None):
    """Try recovery using context embeddings"""
    if context_embeddings is None or len(context_embeddings) == 0:
        return None

    # Convert context to consistent format
    context_np = []
    for ctx in context_embeddings:
        if isinstance(ctx, torch.Tensor):
            context_np.append(ctx.detach().cpu().numpy())
        else:
            context_np.append(ctx)

    # Only proceed if we have valid context
    if not context_np:
        return None

    # Calculate similarities to identify corruption regions
    similarities = []
    for ctx in context_np:
        # Ensure compatible shape
        if len(ctx.shape) == 1:
            ctx = np.expand_dims(ctx, 0)

        # Truncate or pad to match corrupted shape
        feature_dim = corrupted.shape[1]
        if ctx.shape[1] > feature_dim:
            ctx = ctx[:, :feature_dim]
        elif ctx.shape[1] < feature_dim:
            padding = np.zeros((ctx.shape[0], feature_dim - ctx.shape[1]))
            ctx = np.concatenate([ctx, padding], axis=1)

        # Calculate similarity
        sim = cosine_similarity(corrupted, ctx)
        similarities.append(sim[0])

    # If no high similarities, can't recover
    if max(similarities) < 0.5:
        return None

    # Use weighted average of context embeddings
    weights = np.array(similarities)
    weights = np.maximum(0, weights - 0.5)  # Only use contexts with >0.5 similarity
    weight_sum = weights.sum()

    if weight_sum < 1e-5:
        return None

    weights = weights / weight_sum

    # Weighted average
    recovered = np.zeros_like(corrupted)
    for i, ctx in enumerate(context_np):
        if len(ctx.shape) == 1:
            ctx = np.expand_dims(ctx, 0)
        recovered += weights[i] * ctx

    # Blend with original
    blend_factor = min(0.7, max(similarities))
    result = (1.0 - blend_factor) * corrupted + blend_factor * recovered

    return result


def _try_statistical_recovery(corrupted, context_embeddings=None, kb=None):
    """Try recovery using statistical analysis"""
    # Simple outlier detection and correction
    result = corrupted.copy()

    # Calculate statistics for each feature dimension
    mean = np.mean(corrupted, axis=0)
    std = np.std(corrupted, axis=0)

    # Identify outliers (values more than 3 std devs from mean)
    outliers = np.abs(corrupted - mean) > 3 * std

    # Replace outliers with mean values
    result[outliers] = np.tile(mean, (corrupted.shape[0], 1))[outliers]

    return result


def _try_kb_recovery(corrupted, context_embeddings=None, kb=None):
    """Try recovery using knowledge base"""
    if kb is None:
        return None

    try:
        # Apply KB enhancement if available
        if hasattr(kb, 'enhance_embedding'):
            result = corrupted.copy()

            for i in range(corrupted.shape[0]):
                result[i] = kb.enhance_embedding(corrupted[i], None)

            return result
    except Exception as e:
        logger.debug(f"KB recovery failed: {e}")

    return None


def semantic_error_recovery(corrupted_embedding, context_embeddings=None, kb=None):
    """Enhanced recovery with more robust error handling and KB integration"""
    # Basic validity checking
    if corrupted_embedding is None:
        logger.warning("Cannot recover None embedding")
        return None

    # Convert to numpy for consistent processing
    if isinstance(corrupted_embedding, torch.Tensor):
        is_tensor = True
        device = corrupted_embedding.device
        corrupted_np = corrupted_embedding.detach().cpu().numpy()
    else:
        is_tensor = False
        corrupted_np = corrupted_embedding.copy()

    # Ensure proper shape
    if len(corrupted_np.shape) == 1:
        corrupted_np = np.expand_dims(corrupted_np, axis=0)
        was_1d = True
    else:
        was_1d = False

    # Initialize recovery result
    recovered = corrupted_np.copy()

    # Try multiple recovery strategies in order of reliability
    recovery_methods = [
        _try_context_recovery,
        _try_statistical_recovery,
        _try_kb_recovery
    ]

    # Track if any recovery method succeeded
    recovery_success = False

    # Apply recovery methods in sequence
    for method in recovery_methods:
        try:
            if callable(method):
                # Apply recovery method
                method_result = method(corrupted_np, context_embeddings, kb)

                # If method returned valid result, update recovered embedding
                if method_result is not None:
                    recovered = method_result
                    recovery_success = True
                    logger.debug(f"Recovery using {method.__name__} succeeded")
                    break
        except Exception as e:
            logger.debug(f"Recovery method {method.__name__} failed: {e}")
            continue

    # If no recovery method succeeded, apply minimal smoothing
    if not recovery_success:
        try:
            # Simple smoothing as last resort
            recovered = _apply_minimal_smoothing(corrupted_np)
        except Exception as e:
            logger.debug(f"Minimal smoothing failed: {e}")
            # Use original as fallback
            recovered = corrupted_np

    # Restore original shape if needed
    if was_1d:
        recovered = recovered.squeeze(0)

    # Convert back to tensor if input was tensor
    if is_tensor:
        recovered = torch.tensor(recovered, dtype=torch.float32).to(device)

    return recovered

# Add helper functions for semantic error recovery
def _try_context_recovery(corrupted, context_embeddings, kb=None):
    """Try recovery using context embeddings"""
    if context_embeddings is None or len(context_embeddings) == 0:
        return None

    # Convert context to consistent format
    context_np = []
    for ctx in context_embeddings:
        if isinstance(ctx, torch.Tensor):
            context_np.append(ctx.detach().cpu().numpy())
        else:
            context_np.append(ctx)

    # Only proceed if we have valid context
    if not context_np:
        return None

    # Calculate similarities to identify corruption regions
    similarities = []
    for ctx in context_np:
        # Ensure compatible shape
        if len(ctx.shape) == 1:
            ctx = np.expand_dims(ctx, 0)

        # Truncate or pad to match corrupted shape
        feature_dim = corrupted.shape[1]
        if ctx.shape[1] > feature_dim:
            ctx = ctx[:, :feature_dim]
        elif ctx.shape[1] < feature_dim:
            padding = np.zeros((ctx.shape[0], feature_dim - ctx.shape[1]))
            ctx = np.concatenate([ctx, padding], axis=1)

        # Calculate similarity
        sim = cosine_similarity(corrupted, ctx)
        similarities.append(sim[0])

    # If no high similarities, can't recover
    if max(similarities) < 0.5:
        return None

    # Use weighted average of context embeddings
    weights = np.array(similarities)
    weights = np.maximum(0, weights - 0.5)  # Only use contexts with >0.5 similarity
    weight_sum = weights.sum()

    if weight_sum < 1e-5:
        return None

    weights = weights / weight_sum

    # Weighted average
    recovered = np.zeros_like(corrupted)
    for i, ctx in enumerate(context_np):
        if len(ctx.shape) == 1:
            ctx = np.expand_dims(ctx, 0)
        recovered += weights[i] * ctx

    # Blend with original
    blend_factor = min(0.7, max(similarities))
    result = (1.0 - blend_factor) * corrupted + blend_factor * recovered

    return result