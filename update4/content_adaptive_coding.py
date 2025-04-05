import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import json
from tqdm import tqdm
from mlpdvae_utils import ensure_tensor_shape

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import original physical channel
try:
    from physical_channel import PhysicalChannelLayer
except ImportError:
    print("ERROR: Physical channel module not found. Make sure physical_channel.py is in the same directory.")
    raise

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ContentClassifier(nn.Module):
    """Neural network to classify content type from embeddings"""

    def __init__(self, embedding_dim, num_classes=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        # Content class names
        self.content_classes = [
            'procedural',  # Formal parliamentary procedures
            'legislative',  # Legislative discussions and proposals
            'factual',  # Factual statements and reports
            'argumentative'  # Debate and persuasive content
        ]

    def forward(self, x):
        """Classify embedding into content type probability distribution"""
        # Ensure input is proper shape
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        # Get class probabilities
        logits = self.network(x)
        probs = F.softmax(logits, dim=1)
        return probs

    def classify(self, embedding):
        """Classify a single embedding and return content type with dimension adaptation"""
        try:
            # Use utility function to ensure proper shape
            if isinstance(embedding, np.ndarray):
                embedding_tensor = torch.tensor(embedding, dtype=torch.float32).to(device)
            else:
                embedding_tensor = embedding

            # Get the expected input dimension for the classifier
            expected_dim = next(self.network.parameters()).shape[1]

            # Check if dimensions match
            current_dim = embedding_tensor.shape[1] if len(embedding_tensor.shape) > 1 else embedding_tensor.shape[0]

            if current_dim != expected_dim:
                logger.warning(f"Dimension mismatch in classifier: got {current_dim}, expected {expected_dim}")

                # Adapt dimensions - simple approach first
                if len(embedding_tensor.shape) == 1:
                    embedding_tensor = embedding_tensor.unsqueeze(0)

                # Handle dimension mismatch
                if embedding_tensor.shape[1] < expected_dim:
                    # Pad with zeros
                    padding = torch.zeros(embedding_tensor.shape[0], expected_dim - embedding_tensor.shape[1],
                                          device=embedding_tensor.device)
                    embedding_tensor = torch.cat([embedding_tensor, padding], dim=1)
                else:
                    # Truncate
                    embedding_tensor = embedding_tensor[:, :expected_dim]

                logger.info(f"Adapted embedding dimensions from {current_dim} to {expected_dim}")

            # Ensure we have a batch dimension
            if len(embedding_tensor.shape) == 1:
                embedding_tensor = embedding_tensor.unsqueeze(0)

            # Get probabilities
            with torch.no_grad():
                probs = self.forward(embedding_tensor)

            # Get predicted class
            class_idx = torch.argmax(probs, dim=1).item()
            return self.content_classes[class_idx], probs.squeeze().tolist()
        except Exception as e:
            logger.warning(f"Error in content classification: {e}")
            # Return default classification
            return self.content_classes[0], [0.7, 0.1, 0.1, 0.1]


def train_content_classifier(embeddings, sentences, model_save_path='./models/content_classifier.pth', epochs=20,
                             batch_size=32, sample_frac=0.3):
    """
    Train a content classifier on embedding data by identifying content types via keyword analysis.

    Args:
        embeddings: List of embeddings
        sentences: Corresponding sentences
        model_save_path: Where to save the model
        epochs: Number of training epochs
        batch_size: Training batch size
        sample_frac: Fraction of data to use for training (for speed)

    Returns:
        Trained classifier
    """
    # Ensure models directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Prepare data - create labels by analyzing sentence content
    logger.info("Preparing training data by analyzing sentence content")

    # Keywords for identifying content types
    keyword_dict = {
        'procedural': [
            'rule', 'procedure', 'agenda', 'voting', 'session', 'meeting', 'motion',
            'points of order', 'floor', 'president', 'chair', 'adjourn', 'quorum',
            'suspend', 'resume', 'parliament', 'parliamentary', 'assembly'
        ],
        'legislative': [
            'directive', 'regulation', 'decision', 'proposal', 'amendment', 'annex',
            'legislation', 'committee', 'article', 'draft', 'treaty', 'legal',
            'framework', 'law', 'provision', 'act', 'implement', 'vote', 'codecision'
        ],
        'factual': [
            'report', 'statistics', 'data', 'figures', 'study', 'research', 'analysis',
            'survey', 'results', 'findings', 'showed', 'demonstrates', 'according to',
            'evidence', 'experts', 'document', 'published', 'information'
        ],
        'argumentative': [
            'argue', 'position', 'view', 'opinion', 'believe', 'consider', 'debate',
            'disagree', 'oppose', 'support', 'favor', 'against', 'however', 'contrary',
            'whereas', 'nevertheless', 'despite', 'criticize', 'defend', 'advocate'
        ]
    }

    # Assign labels based on keyword matching
    labels = []
    valid_indices = []

    for i, sentence in enumerate(sentences):
        if i >= len(embeddings):
            break

        sentence = sentence.lower()
        scores = {category: 0 for category in keyword_dict.keys()}

        # Count keyword occurrences
        for category, keywords in keyword_dict.items():
            for keyword in keywords:
                if keyword.lower() in sentence:
                    scores[category] += 1

        # Find category with highest score
        max_score = 0
        max_category = 'procedural'  # Default

        for category, score in scores.items():
            if score > max_score:
                max_score = score
                max_category = category

        # Convert category to index
        category_idx = ['procedural', 'legislative', 'factual', 'argumentative'].index(max_category)

        # Only keep examples with clear categorization (at least one keyword match)
        if max_score > 0:
            labels.append(category_idx)
            valid_indices.append(i)

    # Prepare training data
    logger.info(f"Found {len(valid_indices)} valid examples with content labels")

    # Sample a fraction for faster training if needed
    if sample_frac < 1.0:
        sample_size = min(int(len(valid_indices) * sample_frac), len(valid_indices))
        # Randomly sample indices
        sample_indices = np.random.choice(len(valid_indices), size=sample_size, replace=False)
        valid_indices = [valid_indices[idx] for idx in sample_indices]
        labels = [labels[idx] for idx in sample_indices]
        logger.info(f"Sampled {len(valid_indices)} examples for training")

    # Create PyTorch dataset
    X = [embeddings[idx] for idx in valid_indices]
    y = labels

    # Convert to tensors
    X_tensor = torch.tensor(np.array(X), dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)

    # Create classifier
    embedding_dim = X_tensor.shape[1]
    num_classes = 4
    classifier = ContentClassifier(embedding_dim, num_classes).to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    # Create loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    logger.info(f"Training content classifier on {len(X_tensor)} examples")

    # Track metrics
    best_accuracy = 0

    for epoch in range(epochs):
        classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Shuffle indices
        indices = torch.randperm(len(X_tensor))

        # Process in batches
        for i in range(0, len(indices), batch_size):
            # Get batch
            batch_indices = indices[i:i + batch_size]
            inputs = X_tensor[batch_indices]
            targets = y_tensor[batch_indices]

            # Forward pass
            outputs = classifier.network(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        # Calculate epoch metrics
        epoch_loss = running_loss * batch_size / total
        epoch_accuracy = 100 * correct / total

        logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

        # Save best model
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save({
                'model_state_dict': classifier.state_dict(),
                'embedding_dim': embedding_dim,
                'num_classes': num_classes,
                'accuracy': best_accuracy,
                'content_classes': classifier.content_classes
            }, model_save_path)
            logger.info(f"Saved improved model with accuracy {best_accuracy:.2f}%")

    # Final validation
    classifier.eval()
    with torch.no_grad():
        outputs = classifier.network(X_tensor)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = 100 * (predicted == y_tensor).sum().item() / y_tensor.size(0)

    logger.info(f"Final accuracy: {accuracy:.2f}%")
    logger.info(f"Content classifier saved to {model_save_path}")

    # Return trained classifier
    return classifier


class ContentAdaptivePhysicalChannel(PhysicalChannelLayer):
    """
    Enhanced Physical channel layer with content-adaptive coding.
    Extends the original PhysicalChannelLayer with content-aware protection.
    """

    def __init__(self, *args, **kwargs):
        # Extract content-adaptive specific parameters before passing to parent
        self.enable_content_adaptive_coding = kwargs.pop('enable_content_adaptive_coding', True)
        self.content_classifier_path = kwargs.pop('content_classifier_path', './models/content_classifier.pth')

        # Initialize the parent class with remaining parameters
        super().__init__(*args, **kwargs)

        # Add content-adaptive features
        self.content_classifier = None

        # Coding strategies by content type
        self.coding_strategies = {
            'procedural': {
                'good': {'rate': 0.85, 'type': 'repetition', 'interleaving': False},
                'medium': {'rate': 0.75, 'type': 'repetition', 'interleaving': False},
                'poor': {'rate': 0.6, 'type': 'repetition', 'interleaving': True}
            },
            'legislative': {
                'good': {'rate': 0.8, 'type': 'repetition', 'interleaving': False},
                'medium': {'rate': 0.7, 'type': 'repetition', 'interleaving': True},
                'poor': {'rate': 0.5, 'type': 'repetition', 'interleaving': True}
            },
            'factual': {
                'good': {'rate': 0.75, 'type': 'repetition', 'interleaving': False},
                'medium': {'rate': 0.6, 'type': 'repetition', 'interleaving': True},
                'poor': {'rate': 0.4, 'type': 'repetition', 'interleaving': True}
            },
            'argumentative': {
                'good': {'rate': 0.8, 'type': 'repetition', 'interleaving': False},
                'medium': {'rate': 0.7, 'type': 'repetition', 'interleaving': False},
                'poor': {'rate': 0.55, 'type': 'repetition', 'interleaving': True}
            }
        }

        # Importance profile templates by content type - how to weight dimensions
        self.importance_profiles = {
            'procedural': np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]),
            'legislative': np.array([0.8, 0.8, 0.7, 0.7, 0.6, 0.6, 0.5, 0.5, 0.4]),
            'factual': np.array([0.9, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]),
            'argumentative': np.array([0.7, 0.8, 0.9, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        }

        # Try to load content classifier
        self._load_content_classifier()

        logger.info("Content-Adaptive Physical Channel initialized")

    def _load_content_classifier(self):
        """Load the content classifier model if available, with improved error handling"""
        try:
            if os.path.exists(self.content_classifier_path):
                checkpoint = torch.load(self.content_classifier_path, map_location=device)

                # Add validation of checkpoint contents
                if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
                    logger.warning(f"Invalid classifier checkpoint - missing model_state_dict")
                    # Train a basic classifier on the fly if needed
                    return self._train_fallback_classifier()

                # Initialize the model with correct dimensions
                embedding_dim = checkpoint.get('embedding_dim', 768)
                num_classes = checkpoint.get('num_classes', 4)

                self.content_classifier = ContentClassifier(
                    embedding_dim=embedding_dim,
                    num_classes=num_classes
                ).to(device)

                self.content_classifier.load_state_dict(checkpoint['model_state_dict'])
                self.content_classifier.eval()
                logger.info("Content classifier loaded successfully")
                return True
            else:
                # Create a minimal classifier that will work
                return self._train_fallback_classifier()
        except Exception as e:
            logger.error(f"Error loading content classifier: {e}")
            # Create a minimal classifier that will work
            return self._train_fallback_classifier()

    def _train_fallback_classifier(self):
        """Train a minimal classifier with default settings"""
        logger.info("Training fallback content classifier")
        # Create a simple classifier with default settings
        self.content_classifier = ContentClassifier(
            embedding_dim=768,  # Default embedding size
            num_classes=4
        ).to(device)

        # Initialize with random weights but reasonable output distribution
        # This will at least provide valid outputs even if not accurate
        for layer in self.content_classifier.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.7)

        self.content_classifier.eval()
        logger.info("Fallback classifier initialized")
        return True
    def _get_channel_quality(self):
        """Determine channel quality based on SNR and other factors"""
        if self.snr_db > self.snr_threshold_high:
            return 'good'
        elif self.snr_db > self.snr_threshold_low:
            return 'medium'
        else:
            return 'poor'

    def _get_content_adaptive_strategy(self, embedding):
        """Get coding strategy based on content type and channel quality"""
        if not self.enable_content_adaptive_coding or self.content_classifier is None:
            return None, None

        try:
            # Convert to tensor if needed
            if isinstance(embedding, np.ndarray):
                embedding_tensor = torch.tensor(embedding, dtype=torch.float32).to(device)
            else:
                embedding_tensor = embedding

            # Classify content
            content_type, _ = self.content_classifier.classify(embedding_tensor)

            # Get channel quality
            channel_quality = self._get_channel_quality()

            # Get coding strategy
            strategy = self.coding_strategies.get(content_type, {}).get(channel_quality, {})

            # Get importance profile and expand to match embedding dimensions
            if content_type in self.importance_profiles:
                profile_template = self.importance_profiles[content_type]
                # Repeat pattern to match embedding dimension
                repeats = int(np.ceil(len(embedding) / len(profile_template)))
                importance_profile = np.tile(profile_template, repeats)[:len(embedding)]

                # Normalize to [0.2, 1.0] range
                importance_profile = 0.2 + 0.8 * (importance_profile - importance_profile.min()) / \
                                     (importance_profile.max() - importance_profile.min() + 1e-10)
            else:
                # Default uniform profile
                importance_profile = np.ones(len(embedding))

            return strategy, importance_profile

        except Exception as e:
            logger.warning(f"Error in content adaptive strategy: {e}")
            return None, None

    def _apply_content_adaptive_coding(self, bits, embedding):
        """Apply coding with content-adaptive settings"""
        # Get content-based strategy
        strategy, _ = self._get_content_adaptive_strategy(embedding)

        if strategy and self.enable_content_adaptive_coding:
            # Use strategy-specific coding parameters
            original_coding_type = self.coding_type
            original_coding_rate = self.coding_rate

            # Apply strategy settings
            self.coding_type = strategy.get('type', self.coding_type)
            self.coding_rate = strategy.get('rate', self.coding_rate)
            use_interleaving = strategy.get('interleaving', False)

            # Apply coding with temporary settings
            encoded = super()._apply_channel_coding(bits)

            # Apply interleaving if needed
            if use_interleaving:
                encoded = self._apply_interleaving(encoded)

            # Restore original settings
            self.coding_type = original_coding_type
            self.coding_rate = original_coding_rate

            return encoded
        else:
            # Fall back to standard coding
            return super()._apply_channel_coding(bits)

    def _apply_interleaving(self, bits):
        """Apply interleaving to bit stream to distribute burst errors"""
        # Simple block interleaver
        block_size = min(1024, len(bits))
        num_blocks = int(np.ceil(len(bits) / block_size))

        # Pad to fill complete blocks
        padded = np.pad(bits, (0, num_blocks * block_size - len(bits)), 'constant')

        # Reshape and transpose
        blocks = padded.reshape(num_blocks, block_size)
        interleaved = blocks.T.flatten()

        # Trim back to original length
        return interleaved[:len(bits)]

    def _decode_interleaving(self, bits):
        """Reverse the interleaving process"""
        block_size = min(1024, len(bits))
        num_blocks = int(np.ceil(len(bits) / block_size))

        # Pad to fill complete blocks
        padded = np.pad(bits, (0, num_blocks * block_size - len(bits)), 'constant')

        # Reshape and transpose to undo interleaving
        blocks = padded.reshape(block_size, num_blocks)
        deinterleaved = blocks.T.flatten()

        # Trim back to original length
        return deinterleaved[:len(bits)]

    def transmit(self, embedding, importance_weights=None, debug=False):
        """
        Enhanced: Transmit a semantic embedding through the physical channel
        with content-adaptive coding.

        Args:
            embedding: Numpy array or torch tensor containing the semantic embedding
            importance_weights: Optional weights to prioritize important dimensions
            debug: If True, plot constellation diagrams and return extra info

        Returns:
            Received embedding after transmission through the physical channel
        """
        # Convert torch tensor to numpy if needed
        original_tensor = False
        original_device = None
        if isinstance(embedding, torch.Tensor):
            original_tensor = True
            original_device = embedding.device
            embedding = embedding.detach().cpu().numpy()

        # Save original shape for reconstruction
        original_shape = embedding.shape
        flattened = embedding.flatten()

        # Get content-adaptive importance weights if enabled
        if self.enable_content_adaptive_coding and importance_weights is None:
            _, content_importance = self._get_content_adaptive_strategy(flattened)
            if content_importance is not None:
                importance_weights = content_importance

        # Apply importance weighting if provided
        weighted = flattened
        actual_weights = np.ones_like(flattened)

        if self.importance_weighting and importance_weights is not None:
            if isinstance(importance_weights, torch.Tensor):
                importance_weights = importance_weights.detach().cpu().numpy()

            # Reshape weights if needed
            if importance_weights.shape != flattened.shape:
                importance_weights = np.ones_like(flattened)

            # Apply weighting
            weighted = flattened * importance_weights
            actual_weights = importance_weights

        # Convert vector to bits
        bits = self._vector_to_bits(weighted)

        # Apply content-adaptive coding if enabled
        if self.enable_content_adaptive_coding:
            encoded_bits = self._apply_content_adaptive_coding(bits, flattened)
        else:
            # Use standard channel coding
            encoded_bits = super()._apply_channel_coding(bits)

        # Store original bits for error rate estimation
        original_bits = encoded_bits.copy()

        # Map bits to symbols, apply OFDM, and transmit through channel
        # (mostly unchanged from parent class)
        symbols = self._bits_to_symbols(encoded_bits)
        signal = self._apply_ofdm_modulation(symbols)
        received_signal = self._apply_channel_effects(signal)
        received_symbols = self._apply_ofdm_demodulation(received_signal)
        received_bits = self._symbols_to_bits(received_symbols)

        # Check if we used interleaving in content-adaptive mode
        if self.enable_content_adaptive_coding:
            strategy, _ = self._get_content_adaptive_strategy(flattened)
            if strategy and strategy.get('interleaving', False):
                received_bits = self._decode_interleaving(received_bits)

        # Apply channel decoding
        decoded_bits = self._decode_channel_coding(received_bits)

        # Convert bits back to vector
        received_vector = self._bits_to_vector(decoded_bits, original_shape)

        # If importance weighting was applied, unapply it
        if self.importance_weighting and np.any(actual_weights != 1.0):
            # Avoid division by zero
            safe_weights = np.where(actual_weights > 1e-10, actual_weights, 1.0)
            received_vector = received_vector / safe_weights

        # Convert back to tensor if input was a tensor
        if original_tensor:
            received_vector = torch.tensor(received_vector, device=original_device)

        # Return the received embedding and debug info if requested
        if debug:
            debug_info = {
                'bits': bits,
                'encoded_bits': encoded_bits,
                'symbols': symbols,
                'received_symbols': received_symbols,
                'decoded_bits': decoded_bits,
                'error_rate': self._estimate_error_rate(original_bits, received_bits),
                'estimated_snr': self.channel_stats['estimated_snr']
            }

            if self.enable_content_adaptive_coding and self.content_classifier is not None:
                content_type, content_probs = self.content_classifier.classify(embedding)
                debug_info['content_type'] = content_type
                debug_info['content_probs'] = content_probs

            return received_vector, debug_info
        else:
            return received_vector

    def get_channel_info(self):
        """
        Enhanced: Return information about the channel configuration and statistics.
        """
        info = super().get_channel_info()

        # Add content-adaptive information
        info['content_adaptive_coding'] = self.enable_content_adaptive_coding
        info['content_classifier_loaded'] = self.content_classifier is not None

        return info

    def _apply_channel_coding(self, bits, protection_level=1):
        """
        Override parent method to enable content-adaptive coding.
        This will call the _apply_content_adaptive_coding method.
        """
        # Note: this method is kept for backward compatibility
        # The actual adaptive coding is done in _apply_content_adaptive_coding
        return super()._apply_channel_coding(bits, protection_level)


# Function to prepare and train the content classifier
def prepare_content_classifier(data_dir='./data', model_dir='./models', force_train=False):
    """
    Prepare and train content classifier if needed.

    Args:
        data_dir: Directory containing compressed data and sentences
        model_dir: Directory to save model
        force_train: Whether to force training even if model exists

    Returns:
        True if classifier is available, False otherwise
    """
    import pickle

    # Create model directory if needed
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, 'content_classifier.pth')

    # Check if model already exists
    if os.path.exists(model_path) and not force_train:
        logger.info(f"Content classifier already exists at {model_path}")
        return True

    # Load compressed data and sentences
    try:
        # Load sentences
        with open(os.path.join(data_dir, 'processed_data.pkl'), 'rb') as f:
            sentences = pickle.load(f)

        # Load compressed data
        with open(os.path.join(data_dir, 'compressed_data.pkl'), 'rb') as f:
            compressed_data = pickle.load(f)

        # Extract embeddings
        embeddings = []
        for item in compressed_data:
            if isinstance(item, dict) and 'embedding' in item:
                embeddings.append(item['embedding'])
            elif isinstance(item, tuple) and len(item) == 2:
                embeddings.append(item[0])
            else:
                embeddings.append(item)

        # Train classifier
        logger.info(f"Training content classifier on {len(embeddings)} embeddings and {len(sentences)} sentences")
        classifier = train_content_classifier(embeddings, sentences, model_path)

        return True
    except Exception as e:
        logger.error(f"Error preparing content classifier: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


# Example usage when run directly
if __name__ == "__main__":
    import pickle

    print("=== Content Adaptive Coding Module ===")
    print("Preparing content classifier...")

    success = prepare_content_classifier()

    if success:
        print("Content classifier ready!")

        # Test with a sample embedding
        try:
            with open('./data/compressed_data.pkl', 'rb') as f:
                compressed_data = pickle.load(f)

            sample_embedding = compressed_data[0]['embedding'] if isinstance(compressed_data[0], dict) else \
            compressed_data[0]

            # Create content-adaptive channel
            channel = ContentAdaptivePhysicalChannel(
                snr_db=20.0,
                channel_type='awgn',
                modulation='qam',
                modulation_order=16,
                enable_content_adaptive_coding=True
            )

            # Test classification
            content_type, content_probs = channel.content_classifier.classify(sample_embedding)
            print(f"Sample embedding classified as: {content_type}")
            print(f"Probabilities: {[f'{p:.4f}' for p in content_probs]}")

            # Test transmission with content adaptation
            received, debug_info = channel.transmit(sample_embedding, debug=True)
            print(f"Transmission successful, error rate: {debug_info['error_rate']:.4f}")
            print(f"Content type: {debug_info['content_type']}")

        except Exception as e:
            print(f"Error in sample test: {e}")
    else:
        print("Failed to prepare content classifier")