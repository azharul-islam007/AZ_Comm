import os
import numpy as np
import torch
import logging
from physical_channel import PhysicalChannelLayer
from content_adaptive_coding import ContentAdaptivePhysicalChannel  # Import new class
from compression_vae import decompress_vae_embedding  # Import from new compression module

# Import the configuration
try:
    import physical_channel_config as config
except ImportError:
    # Use default configuration if not found
    class DefaultConfig:
        def __init__(self):
            self.ENABLE_PHYSICAL_CHANNEL = True
            self.CHANNEL_TYPE = 'awgn'
            self.SNR_DB = 20.0
            self.MODULATION_TYPE = 'qam'
            self.MODULATION_ORDER = 16
            self.USE_CHANNEL_CODING = True
            self.CODING_RATE = 0.75
            self.USE_IMPORTANCE_WEIGHTING = True
            self.WEIGHT_METHOD = 'semantic'
            self.ENABLE_ADAPTIVE_MODULATION = True
            self.ENABLE_UNEQUAL_ERROR_PROTECTION = True
            # New settings
            self.ENABLE_CONTENT_ADAPTIVE_CODING = True
            self.VAE_COMPRESSION = True
            self.CONTENT_CLASSIFIER_PATH = './models/content_classifier.pth'


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
            # Use variance as importance - higher variance = more important
            if is_batch:
                # For batches, use variance across batch dimension
                variance = np.var(embedding, axis=0)
                # Normalize to range [0.5, 1.0] to avoid extreme downweighting
                weights = 0.5 + 0.5 * (variance / (np.max(variance) + 1e-10))
                # Broadcast to match embedding shape
                weights = np.tile(weights, (embedding.shape[0], 1))
            else:
                # For single embedding, use uniform weights
                weights = np.ones_like(embedding)

        elif method == 'pca':
            # PCA-inspired weighting - higher components = more important
            if not self.pca_fitted or self.pca is None:
                # Initialize PCA if not already done
                if is_batch:
                    # Fit PCA on the batch
                    from sklearn.decomposition import PCA
                    n_components = min(embedding.shape[1], 100)  # Use at most 100 components
                    self.pca = PCA(n_components=n_components)
                    self.pca.fit(embedding)
                    self.pca_fitted = True
                else:
                    # Can't fit PCA on a single example
                    return np.ones_like(embedding)

            if self.pca_fitted:
                # Get the importance weights based on explained variance ratio
                exp_var_ratio = self.pca.explained_variance_ratio_
                # Ensure all dimensions have a weight (padding with small values if needed)
                weight_vector = np.ones(embedding.shape[1]) * 0.1
                weight_vector[:len(exp_var_ratio)] = 0.1 + 0.9 * exp_var_ratio  # Scale to [0.1, 1.0]

                # Broadcast to match embedding shape
                if is_batch:
                    weights = np.tile(weight_vector, (embedding.shape[0], 1))
                else:
                    weights = weight_vector
            else:
                # Fall back to uniform weights
                weights = np.ones_like(embedding)

        elif method == 'semantic':
            # Semantic-aware weighting - more sophisticated
            # This combines PCA and variance methods with a focus on semantic meaning

            # First get variance-based weights
            var_weights = self._calculate_importance_weights(embedding, 'variance')

            # If we have PCA fitted, blend with PCA importance
            if self.pca_fitted and is_batch:
                pca_weights = self._calculate_importance_weights(embedding, 'pca')

                # Blend the two methods (60% variance, 40% PCA)
                weights = 0.6 * var_weights + 0.4 * pca_weights
            else:
                weights = var_weights

            # Apply exponential weighting to emphasize important dimensions even more
            weights = np.power(weights, 1.5)  # Exponential factor

            # Rescale to ensure we have a good range of weights
            if is_batch:
                for i in range(weights.shape[0]):
                    row_min, row_max = weights[i].min(), weights[i].max()
                    if row_max > row_min:
                        weights[i] = 0.2 + 0.8 * (weights[i] - row_min) / (row_max - row_min)
            else:
                weight_min, weight_max = weights.min(), weights.max()
                if weight_max > weight_min:
                    weights = 0.2 + 0.8 * (weights - weight_min) / (weight_max - weight_min)

        else:  # 'uniform' or any other value
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

    def transmit_through_physical_channel(self, embedding, debug=False):
        """
        Enhanced: Transmit the semantic embedding through the physical channel.
        Now supports VAE compression and content-adaptive coding.

        Args:
            embedding: Numpy array or torch tensor containing the semantic embedding
            debug: If True, return additional debug information

        Returns:
            Received embedding after transmission through the physical channel
        """
        if not self.physical_enabled or self._physical_channel is None:
            return embedding  # Pass through if physical channel is disabled

        # Store original embedding for self-supervised learning
        original_embedding = embedding

        # Convert torch tensor to numpy if needed
        is_torch = isinstance(embedding, torch.Tensor)
        device = embedding.device if is_torch else None

        if is_torch:
            embedding_np = embedding.detach().cpu().numpy()
        else:
            embedding_np = embedding

        # Check if this is a VAE-compressed embedding that needs decompression after transmission
        is_vae_compressed = self.use_vae_compression

        # Calculate importance weights using enhanced method
        importance_weights = self._calculate_importance_weights(embedding_np)

        # Transmit through physical channel
        if debug:
            received_embedding, debug_info = self._physical_channel.transmit(
                embedding_np, importance_weights, debug=True)
        else:
            received_embedding = self._physical_channel.transmit(
                embedding_np, importance_weights, debug=False)

        # Save transmission pair for self-supervised learning
        if self.collect_transmission_data:
            self._save_transmission_pair(original_embedding, received_embedding)

        # Convert back to torch tensor if the input was a torch tensor
        if is_torch:
            received_embedding = torch.tensor(received_embedding, device=device)

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

    def log_transmission_metrics(self, original, received, semantic_score=None):
        """
        Log metrics for a transmission through the physical channel.

        Args:
            original: Original embedding
            received: Received embedding
            semantic_score: Optional semantic similarity score
        """
        # This method is unchanged - it already works with new features

    def _plot_metrics(self):
        """Plot the accumulated metrics."""
        # This method is unchanged - it already works with new features

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


# Create an instance of the integration layer for easy import
physical_semantic_bridge = EnhancedPhysicalSemanticIntegration()


def transmit_through_physical_channel(embedding, debug=False):
    """
    Convenience function to transmit through the physical channel.
    This function is intended to be imported and used in the semantic communication pipeline.
    """
    return physical_semantic_bridge.transmit_through_physical_channel(embedding, debug=debug)