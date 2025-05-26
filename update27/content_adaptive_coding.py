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


class EnhancedContentClassifier(nn.Module):
    """Neural network to classify content type from embeddings"""

    def __init__(self, embedding_dim=460, num_classes=5):  # Expanded classes
        super().__init__()
        self.embedding_dim = embedding_dim

        # Dimension adapter
        self.dimension_adapter = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU()
        )

        # Self-attention layer
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Main network with deeper architecture
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

        # Expanded content class definitions
        self.content_classes = [
            'procedural',  # Formal parliamentary procedures
            'legislative',  # Legislative discussions and proposals
            'factual',  # Factual statements and reports
            'argumentative',  # Debate and persuasive content
            'administrative'  # Administrative and organizational content
        ]

    def forward(self, x):
        """Forward pass with attention mechanism"""
        # Ensure proper shape
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        # Apply dimension adapter
        x = self.dimension_adapter(x)

        # Apply attention
        attention_weights = self.attention(x)
        attention_weights = F.softmax(attention_weights, dim=1)

        # Apply attention to input
        attended_input = x * attention_weights

        # Get class probabilities
        logits = self.network(attended_input)
        probs = F.softmax(logits, dim=1)

        return probs

    def classify(self, embedding):
        """Classify a single embedding and return content type with improved dimension handling"""
        try:
            # Convert to tensor if needed
            if isinstance(embedding, np.ndarray):
                embedding_tensor = torch.tensor(embedding, dtype=torch.float32).to(device)
            else:
                embedding_tensor = embedding.clone().detach()

            # Ensure embedding is on the device
            embedding_tensor = embedding_tensor.to(device)

            # Ensure we have a batch dimension
            if len(embedding_tensor.shape) == 1:
                embedding_tensor = embedding_tensor.unsqueeze(0)

            # Get the expected input dimension
            expected_dim = self.embedding_dim
            current_dim = embedding_tensor.shape[1]

            # NEW: Log dimension information for debugging
            if current_dim != expected_dim:
                logger.debug(f"Content classifier input dimension: {current_dim}, expected: {expected_dim}")

            # Handle dimension mismatch using improved method
            if current_dim != expected_dim:
                embedding_tensor = self._resize_input_tensor(embedding_tensor, expected_dim)

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

    # ADD this new method to ContentClassifier class:
    def _resize_input_tensor(self, tensor, target_dim):
        """Resize input tensor to match expected dimensions using adaptive approach"""
        current_dim = tensor.shape[1]

        if current_dim < target_dim:
            # If input is smaller, pad with zeros
            padding = torch.zeros(tensor.shape[0], target_dim - current_dim, device=tensor.device)
            return torch.cat([tensor, padding], dim=1)
        else:
            # If input is larger, use a combination of averaging and truncation
            ratio = current_dim / target_dim
            if ratio > 2.0:
                # For very large ratios, use average pooling to preserve information
                reshaped = tensor.view(tensor.shape[0], 1, current_dim)
                pooled = F.avg_pool1d(reshaped, kernel_size=int(ratio), stride=int(ratio))
                # Resize to ensure exact dimensions
                return F.interpolate(pooled, size=target_dim).view(tensor.shape[0], target_dim)
            else:
                # For smaller ratios, use simple truncation
                return tensor[:, :target_dim]

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
    classifier = EnhancedContentClassifier(embedding_dim, num_classes).to(device)

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
        self.embedding_dim = kwargs.pop('embedding_dim', 768)  # Extract embedding dimension parameter

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
        self._last_retransmission_count = 0
        self._total_retransmissions = 0
        self._arq_triggers = {
            'crc_failure': 0,
            'semantic_anchor_corruption': 0,
            'probabilistic_semantic': 0
        }

        logger.info("Smart-ARQ and Semantic-Aware FEC enabled")

    # In content_adaptive_coding.py, modify _load_content_classifier method:
    def _load_content_classifier(self):
        """
        Load content classifier with improved error handling for architecture mismatches.
        Returns True if successful, False otherwise.
        """
        try:
            # Detect embedding dimension first
            embedding_dim = self.embedding_dim

            # Try loading the checkpoint first to get the correct embedding dimension
            if os.path.exists(self.content_classifier_path):
                checkpoint = torch.load(self.content_classifier_path, map_location=device)

                # Use the dimension from the checkpoint if available
                if isinstance(checkpoint, dict) and 'embedding_dim' in checkpoint:
                    checkpoint_dim = checkpoint.get('embedding_dim')
                    if checkpoint_dim is not None:
                        # Use the checkpoint's dimension to ensure compatibility
                        embedding_dim = checkpoint_dim
                        logger.info(f"Using embedding dimension {embedding_dim} from checkpoint")

            # Create model with correct dimensions from checkpoint
            self.content_classifier = EnhancedContentClassifier(
                embedding_dim=embedding_dim,
                num_classes=4
            ).to(device)

            # Try loading weights if file exists, with improved error handling
            if os.path.exists(self.content_classifier_path):
                try:
                    # First check if the model architecture is fully compatible
                    strict_load_success = False
                    try:
                        self.content_classifier.load_state_dict(checkpoint['model_state_dict'], strict=True)
                        strict_load_success = True
                        logger.info("Content classifier loaded with strict parameter matching")
                    except Exception as e:
                        # If strict loading fails, try non-strict loading
                        logger.debug(f"Strict model loading failed: {e}")

                    if not strict_load_success:
                        try:
                            # Try with non-strict loading to handle architecture differences
                            self.content_classifier.load_state_dict(checkpoint['model_state_dict'], strict=False)
                            logger.info(
                                "Content classifier loaded with partial parameter matching (architecture mismatch)")
                        except Exception as e:
                            # If even non-strict loading fails, use the fallback method
                            logger.warning(f"Could not load classifier weights even with partial matching: {e}")
                            self._initialize_fallback_classifier(embedding_dim)
                except Exception as e:
                    logger.warning(f"Could not load classifier weights: {e}")
                    # Use improved fallback method with correct dimension
                    self._initialize_fallback_classifier(embedding_dim)
            else:
                # Initialize with fallback if no checkpoint exists
                self._initialize_fallback_classifier(embedding_dim)

            self.content_classifier.eval()
            return True

        except Exception as e:
            logger.error(f"Error in content classifier initialization: {e}")
            return False

    def get_arq_statistics(self):
        """Get Smart-ARQ usage statistics"""
        return {
            'total_retransmissions': self._total_retransmissions,
            'trigger_reasons': self._arq_triggers.copy(),
            'last_retransmission_count': self._last_retransmission_count
        }

    def _initialize_default_weights(self):
        """Initialize classifier with reasonable default weights"""
        for m in self.content_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _initialize_fallback_classifier(self, embedding_dim=None):
        """Initialize fallback content classifier with a functional model"""
        # Make sure we have an embedding dimension
        if embedding_dim is None:
            if hasattr(self, 'embedding_dim'):
                embedding_dim = self.embedding_dim
            else:
                embedding_dim = 460  # Default size if all else fails

        # Create a fresh classifier
        self.content_classifier = EnhancedContentClassifier(
            embedding_dim=embedding_dim,
            num_classes=4
        ).to(device)

        # Initialize with reasonable random weights
        for m in self.content_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Save the new classifier to avoid future loading issues
        model_dir = os.path.dirname(self.content_classifier_path)
        os.makedirs(model_dir, exist_ok=True)

        # Save the initialized model
        torch.save({
            'model_state_dict': self.content_classifier.state_dict(),
            'embedding_dim': embedding_dim,
            'num_classes': 4,
            'content_classes': self.content_classifier.content_classes
        }, self.content_classifier_path)

        logger.info(f"Created and saved new content classifier with embedding_dim={embedding_dim}")

        # Set to eval mode
        self.content_classifier.eval()

    def _apply_semantic_aware_fec(self, bits, embedding, importance_weights):
        """Apply semantic-aware FEC with overprotection for salient bits"""
        if importance_weights is None:
            return super()._apply_channel_coding(bits)

        # Identify salient regions with more aggressive thresholds
        threshold_high = np.percentile(importance_weights, 85)  # Top 15%
        threshold_med = np.percentile(importance_weights, 60)  # Next 25%

        high_importance = importance_weights >= threshold_high
        med_importance = (importance_weights >= threshold_med) & (importance_weights < threshold_high)
        low_importance = importance_weights < threshold_med

        # Map to bit indices
        bits_per_dim = len(bits) // len(importance_weights)

        # Collect bits by importance level
        high_bits = []
        med_bits = []
        low_bits = []
        high_indices = []
        med_indices = []
        low_indices = []

        for i, weight in enumerate(importance_weights):
            start_bit = i * bits_per_dim
            end_bit = min((i + 1) * bits_per_dim, len(bits))
            bit_range = list(range(start_bit, end_bit))

            if high_importance[i]:
                high_bits.extend(bits[start_bit:end_bit])
                high_indices.extend(bit_range)
            elif med_importance[i]:
                med_bits.extend(bits[start_bit:end_bit])
                med_indices.extend(bit_range)
            else:
                low_bits.extend(bits[start_bit:end_bit])
                low_indices.extend(bit_range)

        # Apply different FEC rates - OVERPROTECT salient bits
        original_rate = self.coding_rate

        # High importance: Very strong protection (rate 0.3)
        if high_bits:
            self.coding_rate = 0.3
            encoded_high = super()._apply_channel_coding(np.array(high_bits))
        else:
            encoded_high = np.array([])

        # Medium importance: Strong protection (rate 0.5)
        if med_bits:
            self.coding_rate = 0.5
            encoded_med = super()._apply_channel_coding(np.array(med_bits))
        else:
            encoded_med = np.array([])

        # Low importance: Standard protection (rate 0.8)
        if low_bits:
            self.coding_rate = 0.8
            encoded_low = super()._apply_channel_coding(np.array(low_bits))
        else:
            encoded_low = np.array([])

        # Restore original rate
        self.coding_rate = original_rate

        # Reassemble encoded bits in original order
        total_encoded_length = len(encoded_high) + len(encoded_med) + len(encoded_low)
        encoded_bits = np.zeros(total_encoded_length, dtype=bits.dtype)

        # Simple reassembly - in practice you'd want more sophisticated interleaving
        high_len = len(encoded_high)
        med_len = len(encoded_med)

        if high_len > 0:
            encoded_bits[:high_len] = encoded_high
        if med_len > 0:
            encoded_bits[high_len:high_len + med_len] = encoded_med
        if len(encoded_low) > 0:
            encoded_bits[high_len + med_len:] = encoded_low

        logger.debug(f"Semantic FEC: High={len(high_bits)}, Med={len(med_bits)}, Low={len(low_bits)} bits")

        return encoded_bits
    def _get_channel_quality(self):
        """Determine channel quality based on SNR and other factors"""
        if self.snr_db > self.snr_threshold_high:
            return 'good'
        elif self.snr_db > self.snr_threshold_low:
            return 'medium'
        else:
            return 'poor'

    def _get_content_adaptive_strategy(self, embedding):
        """Get optimized coding strategy based on content and SNR with enhanced protection"""
        if not self.enable_content_adaptive_coding or self.content_classifier is None:
            return None, None

        try:
            # Get channel quality with more aggressive SNR thresholds
            channel_quality = self._get_channel_quality()

            # Get content classification with error handling
            detected_content_type = "factual"  # Default if classification fails
            confidence = 0.5  # Default medium confidence

            try:
                # Classify content
                detected_content_type, probs = self.content_classifier.classify(embedding)
                # Get highest probability as confidence
                confidence = max(probs) if isinstance(probs, list) else 0.5
            except Exception as e:
                logger.debug(f"Content classification failed: {e}")

            # Higher confidence = more aggressive optimization
            confidence_factor = min(1.0, confidence * 1.2)  # Boost confidence slightly

            # Get FEC rates from config
            from config_manager import ConfigManager
            config = ConfigManager()

            # SNR-based FEC rates (more aggressive protection)
            if self.snr_db >= 20.0:
                base_rate = config.get("physical.fec_buckets.>=20", 0.8)
            elif self.snr_db >= 15.0:
                base_rate = config.get("physical.fec_buckets.15-20", 0.6)
            elif self.snr_db >= 10.0:
                base_rate = config.get("physical.fec_buckets.10-15", 0.4)
            else:
                base_rate = config.get("physical.fec_buckets.<10", 0.2)

            # Adjust rate based on content type
            if detected_content_type == "procedural":
                # Procedural content (rules, procedures) gets stronger protection
                rate = base_rate * 0.9 * confidence_factor
                use_interleaving = channel_quality != 'good'
                use_ldpc = channel_quality != 'good'
            elif detected_content_type == "legislative":
                # Legislative content also needs higher protection
                rate = base_rate * 0.95 * confidence_factor
                use_interleaving = channel_quality != 'good'
                use_ldpc = channel_quality != 'good'
            else:
                # Default strategy for other content types - less critical
                rate = base_rate * confidence_factor
                use_interleaving = channel_quality == 'poor'
                use_ldpc = channel_quality == 'poor'

            # Create strategy object
            strategy = {
                'rate': max(0.2, min(0.9, rate)),  # More aggressive clamping (0.2-0.9)
                'type': 'repetition',
                'interleaving': use_interleaving,
                'use_ldpc': use_ldpc or channel_quality == 'poor',  # More aggressive LDPC usage
            }

            # Get importance profile for this content type
            if detected_content_type in self.importance_profiles:
                profile_template = self.importance_profiles[detected_content_type]
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
            logger.warning(f"Error in adaptive strategy selection: {e}")
            return None, None

    def _apply_ldpc_coding(self, bits):
        """Apply LDPC coding for more robust error protection"""
        # Simple parity check coding as a lightweight LDPC approximation
        # In practice, you would use a proper LDPC library
        coded_bits = []

        # Process in blocks of 8 bits
        for i in range(0, len(bits), 8):
            block = bits[i:i + 8]

            # Calculate parity bits (4 parity bits for 8 data bits)
            parity1 = block[0] ^ block[1] ^ block[2] ^ block[3]
            parity2 = block[0] ^ block[4] ^ block[5] ^ block[6]
            parity3 = block[1] ^ block[3] ^ block[5] ^ block[7]
            parity4 = block[2] ^ block[3] ^ block[6] ^ block[7]

            # Add data and parity bits
            coded_block = np.concatenate([block, [parity1, parity2, parity3, parity4]])
            coded_bits.extend(coded_block)

        return np.array(coded_bits)

    def _decode_ldpc(self, bits):
        """Decode LDPC-coded bits"""
        decoded_bits = []

        # Process in blocks of 12 bits (8 data + 4 parity)
        for i in range(0, len(bits), 12):
            block = bits[i:i + 12]

            # If block is incomplete, return what we have
            if len(block) < 12:
                decoded_bits.extend(block[:min(8, len(block))])
                continue

            # Extract data and parity bits
            data = block[:8]
            parity = block[8:12]

            # Calculate expected parity bits
            expected_parity1 = data[0] ^ data[1] ^ data[2] ^ data[3]
            expected_parity2 = data[0] ^ data[4] ^ data[5] ^ data[6]
            expected_parity3 = data[1] ^ data[3] ^ data[5] ^ data[7]
            expected_parity4 = data[2] ^ data[3] ^ data[6] ^ data[7]

            # Check if parity matches
            parity_ok = (parity[0] == expected_parity1 and
                         parity[1] == expected_parity2 and
                         parity[2] == expected_parity3 and
                         parity[3] == expected_parity4)

            # If parity doesn't match, try to correct errors
            # This is a simplified correction - real LDPC would be more sophisticated
            if not parity_ok:
                # Check which parity bits are wrong
                errors = [parity[0] != expected_parity1,
                          parity[1] != expected_parity2,
                          parity[2] != expected_parity3,
                          parity[3] != expected_parity4]

                # Simple error correction for single bit errors
                if sum(errors) == 1:
                    # Single parity error could indicate error in the parity bit itself
                    pass  # Just keep the data as is
                elif sum(errors) == 2:
                    # Two parity errors could indicate a data bit error
                    # Real LDPC would do better correction here
                    pass

            # Add the data bits
            decoded_bits.extend(data)

        return np.array(decoded_bits)

    def _apply_content_adaptive_coding(self, bits, embedding):
        """Apply coding with content-adaptive settings and LDPC for critical dimensions"""
        # Get content-based strategy
        strategy, importance_weights = self._get_content_adaptive_strategy(embedding)

        if strategy and self.enable_content_adaptive_coding:
            # Use strategy-specific coding parameters
            original_coding_type = self.coding_type
            original_coding_rate = self.coding_rate

            # Apply strategy settings
            self.coding_type = strategy.get('type', self.coding_type)
            self.coding_rate = strategy.get('rate', self.coding_rate)
            use_interleaving = strategy.get('interleaving', False)
            use_ldpc = strategy.get('use_ldpc', False)

            # Identify critical dimensions (top 25%) if importance weights available
            if importance_weights is not None:
                # Flatten importance weights if needed
                importance_weights = importance_weights.flatten() if hasattr(importance_weights,
                                                                             'flatten') else importance_weights

                # Get indices of bits from most important dimensions
                critical_threshold = np.percentile(importance_weights, 75)
                critical_indices = np.where(importance_weights >= critical_threshold)[0]

                # Calculate bit indices (assuming each dimension has multiple bits)
                bits_per_dim = len(bits) // len(importance_weights)
                critical_bit_indices = []

                for idx in critical_indices:
                    start = idx * bits_per_dim
                    end = (idx + 1) * bits_per_dim
                    critical_bit_indices.extend(range(start, min(end, len(bits))))

                # Split bits into critical and non-critical
                critical_bits = bits[critical_bit_indices]
                non_critical_indices = np.setdiff1d(np.arange(len(bits)), critical_bit_indices)
                non_critical_bits = bits[non_critical_indices]

                # Apply stronger coding (LDPC or lower rate) to critical bits
                if use_ldpc and len(critical_bits) > 0:
                    # Apply LDPC to critical bits
                    encoded_critical = self._apply_ldpc_coding(critical_bits)
                else:
                    # Apply stronger repetition code to critical bits
                    original_rate = self.coding_rate
                    self.coding_rate = max(0.5, self.coding_rate * 0.8)  # Stronger protection
                    encoded_critical = super()._apply_channel_coding(critical_bits)
                    self.coding_rate = original_rate

                # Standard coding for non-critical bits
                encoded_non_critical = super()._apply_channel_coding(non_critical_bits)

                # Reassemble in original order
                encoded = np.zeros(len(encoded_critical) + len(encoded_non_critical), dtype=bits.dtype)
                critical_output_indices = []
                non_critical_output_idx = 0

                # Build mapping for critical bits in output
                for i in range(len(bits)):
                    if i in critical_bit_indices:
                        critical_idx = np.where(critical_bit_indices == i)[0][0]
                        critical_bits_start = critical_idx * (len(encoded_critical) // len(critical_bits))
                        critical_bits_end = (critical_idx + 1) * (len(encoded_critical) // len(critical_bits))
                        critical_output_indices.extend(range(critical_bits_start, critical_bits_end))
                    else:
                        # For non-critical bits
                        non_critical_bits_count = len(encoded_non_critical) // len(non_critical_bits)
                        encoded[non_critical_output_idx:non_critical_output_idx + non_critical_bits_count] = \
                            encoded_non_critical[
                            non_critical_output_idx:non_critical_output_idx + non_critical_bits_count]
                        non_critical_output_idx += non_critical_bits_count

                # Place critical bits in output
                for i, idx in enumerate(critical_output_indices):
                    if idx < len(encoded) and i < len(encoded_critical):
                        encoded[idx] = encoded_critical[i]
            else:
                # Without importance weights, encode everything with the strategy's parameters
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

    def apply_content_adaptive_modulation(self, bits, importance_weights):
        """
        Apply content-adaptive modulation based on importance weights.

        Args:
            bits: The bits to modulate
            importance_weights: Importance weights for each dimension

        Returns:
            Modulated symbols
        """
        # Flatten importance weights if needed
        if isinstance(importance_weights, torch.Tensor):
            importance_weights = importance_weights.detach().cpu().numpy()

        if len(importance_weights.shape) > 1:
            importance_weights = importance_weights.flatten()

        # Ensure weights match bits length
        if len(importance_weights) < len(bits):
            # Repeat weights to match bits length
            repetitions = int(np.ceil(len(bits) / len(importance_weights)))
            importance_weights = np.tile(importance_weights, repetitions)[:len(bits)]

        # Segment bits by importance
        high_imp_mask = importance_weights >= 0.7
        med_imp_mask = (importance_weights >= 0.4) & (importance_weights < 0.7)
        low_imp_mask = importance_weights < 0.4

        # Get channel conditions
        current_snr = self.channel_stats['estimated_snr']

        # Adjust modulation based on importance and SNR
        if current_snr > 25:  # Excellent conditions
            # Use QAM-64 for low importance, QAM-16 for medium, QPSK for high
            high_imp_order = 4  # QPSK
            med_imp_order = 16  # QAM-16
            low_imp_order = 64  # QAM-64
        elif current_snr > 15:  # Good conditions
            # Use QAM-16 for low importance, QPSK for medium/high
            high_imp_order = 4  # QPSK
            med_imp_order = 4  # QPSK
            low_imp_order = 16  # QAM-16
        else:  # Poor conditions
            # Use QPSK for all - safety first
            high_imp_order = 4  # QPSK
            med_imp_order = 4  # QPSK
            low_imp_order = 4  # QPSK

        # Initialize modulated symbols array
        symbols_count = len(bits) // 2  # Assuming at least QPSK (2 bits/symbol)
        symbols = np.zeros(symbols_count, dtype=complex)

        # Store original modulation parameters
        original_modulation_order = self.modulation_order

        try:
            # Process high importance bits with QPSK
            if np.any(high_imp_mask):
                high_bits = bits[high_imp_mask]
                self.modulation_order = high_imp_order
                self._init_constellation()  # Reinitialize constellation
                high_symbols = self._bits_to_symbols(high_bits)

                # Place in symbol array (simplified - actual implementation would be more complex)
                high_indices = np.where(high_imp_mask)[0] // 2
                high_indices = high_indices[:len(high_symbols)]
                for i, idx in enumerate(high_indices):
                    if idx < len(symbols):
                        symbols[idx] = high_symbols[i % len(high_symbols)]

            # Restore original modulation
            self.modulation_order = original_modulation_order
            self._init_constellation()

            return symbols
        except Exception as e:
            # Restore original modulation in case of error
            self.modulation_order = original_modulation_order
            self._init_constellation()
            logger.warning(f"Error in content adaptive modulation: {e}")

            # Fall back to standard modulation
            return self._bits_to_symbols(bits)

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

    def _check_frame_integrity(self, original_bits, received_bits, importance_weights=None):
        """Check frame integrity and identify corruption in salient regions"""
        # Calculate simple bit error rate
        min_len = min(len(original_bits), len(received_bits))
        if min_len == 0:
            return True, False

        errors = np.sum(original_bits[:min_len] != received_bits[:min_len])
        ber = errors / min_len

        # Frame fails if BER > 10%
        frame_ok = ber <= 0.1

        # Check for semantic anchor corruption
        semantic_corruption = False
        if importance_weights is not None and len(importance_weights) > 0:
            # Map importance weights to bit indices
            bits_per_dim = len(original_bits) // len(importance_weights)

            # Find high-importance regions (top 25%)
            threshold = np.percentile(importance_weights, 75)
            critical_indices = np.where(importance_weights >= threshold)[0]

            # Check for errors in critical regions
            critical_errors = 0
            total_critical_bits = 0

            for dim_idx in critical_indices:
                start_bit = dim_idx * bits_per_dim
                end_bit = min((dim_idx + 1) * bits_per_dim, len(original_bits))

                if end_bit <= len(received_bits):
                    dim_errors = np.sum(original_bits[start_bit:end_bit] != received_bits[start_bit:end_bit])
                    critical_errors += dim_errors
                    total_critical_bits += (end_bit - start_bit)

            # Flag semantic corruption if >5% of critical bits are wrong
            if total_critical_bits > 0:
                semantic_corruption = (critical_errors / total_critical_bits) > 0.05

        return frame_ok, semantic_corruption

    def _smart_arq_decision(self, frame_ok, semantic_corruption, content_type="unknown"):
        """Decide whether to trigger Smart-ARQ retransmission"""
        # Always retransmit if frame failed
        if not frame_ok:
            return True, "crc_failure"

        # For critical content types, also retransmit on semantic corruption
        critical_content = content_type in ["procedural", "legislative"]

        if semantic_corruption and critical_content:
            return True, "semantic_anchor_corruption"

        # Check if we should do probabilistic retransmission for high corruption
        if semantic_corruption and np.random.random() < 0.3:  # 30% chance
            return True, "probabilistic_semantic"

        return False, "no_retransmission"

    def transmit(self, embedding, importance_weights=None, debug=False, max_retransmissions=2):
        """Enhanced transmit with Smart-ARQ"""
        # Get content classification first
        content_type = "unknown"
        if self.content_classifier:
            try:
                content_type, _ = self.content_classifier.classify(embedding)
            except:
                pass

        retransmission_count = 0
        received_vector = None  # Initialize to avoid unresolved reference

        # Convert embedding to numpy format once
        if isinstance(embedding, torch.Tensor):
            embedding_np = embedding.detach().cpu().numpy()
            is_tensor = True
            original_device = embedding.device
            original_dtype = embedding.dtype
        else:
            embedding_np = embedding
            is_tensor = False
            original_device = None
            original_dtype = None

        if len(embedding_np.shape) == 1:
            embedding_np = np.expand_dims(embedding_np, 0)
            was_1d = True
        else:
            was_1d = False

        original_shape = embedding_np.shape
        flattened = embedding_np.flatten()

        # Calculate importance weights if not provided
        if importance_weights is None:
            _, importance_weights = self._get_content_adaptive_strategy(flattened)
            if importance_weights is None:
                importance_weights = np.ones_like(flattened)

        while retransmission_count <= max_retransmissions:
            # Convert to bits
            bits = self._vector_to_bits(flattened)
            original_bits = bits.copy()

            # Apply semantic-aware FEC
            encoded_bits = self._apply_semantic_aware_fec(bits, flattened, importance_weights)

            # Transmit through channel
            symbols = self._bits_to_symbols(encoded_bits)
            signal = self._apply_ofdm_modulation(symbols)
            received_signal = self._apply_channel_effects(signal)
            received_symbols = self._apply_ofdm_demodulation(received_signal)
            received_bits = self._symbols_to_bits(received_symbols)

            # Decode
            decoded_bits = self._decode_channel_coding(received_bits)

            # Convert back to vector
            received_vector = self._bits_to_vector(decoded_bits, original_shape)

            # Restore original shape if needed
            if was_1d:
                received_vector = received_vector.squeeze(0)

            # Convert back to tensor if input was tensor
            if is_tensor:
                received_vector = torch.tensor(received_vector, device=original_device, dtype=original_dtype)

            # Check frame integrity and semantic corruption
            frame_ok, semantic_corruption = self._check_frame_integrity(
                original_bits, decoded_bits[:len(original_bits)], importance_weights)

            # Smart-ARQ decision
            should_retransmit, reason = self._smart_arq_decision(
                frame_ok, semantic_corruption, content_type)

            # Update statistics
            self._last_retransmission_count = retransmission_count
            if should_retransmit and retransmission_count < max_retransmissions:
                self._total_retransmissions += 1
                if reason in self._arq_triggers:
                    self._arq_triggers[reason] += 1

            if not should_retransmit or retransmission_count == max_retransmissions:
                # Accept this transmission
                if retransmission_count > 0:
                    logger.info(f"Smart-ARQ: Accepted after {retransmission_count} retransmissions")

                return received_vector
            else:
                # Retransmit
                retransmission_count += 1
                logger.info(f"Smart-ARQ: Retransmission #{retransmission_count} due to {reason}")

                # Optionally adjust parameters for retransmission
                if reason == "semantic_anchor_corruption":
                    # Use even stronger protection for retransmission
                    original_rate = self.coding_rate
                    self.coding_rate = max(0.2, self.coding_rate - 0.1)
                    logger.debug(f"Adjusted coding rate from {original_rate} to {self.coding_rate} for retransmission")

        # This should not be reached due to loop logic, but included for safety
        logger.warning("Smart-ARQ: Maximum retransmissions reached, returning last attempt")
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


def train_enhanced_content_classifier(embeddings, sentences,
                                      model_save_path='./models/enhanced_content_classifier.pth'):
    """Train enhanced content classifier with better labeling and more data"""
    import re

    # Create model
    embedding_dim = embeddings[0].shape[0]
    classifier = EnhancedContentClassifier(embedding_dim=embedding_dim).to(device)
    # Get number of classes from classifier
    num_classes = len(classifier.content_classes)

    # Enhanced labeling function
    def assign_content_label(text):
        """Assign content label with more sophisticated rules"""
        # Define category keywords with weights
        category_keywords = {
            'procedural': {
                'agenda': 2.0, 'rule': 2.0, 'point of order': 3.0, 'procedure': 2.0,
                'session': 1.5, 'adjourn': 2.0, 'vote': 1.5, 'motion': 1.5,
                'chair': 1.0, 'president': 1.0, 'floor': 1.0
            },
            'legislative': {
                'directive': 2.0, 'regulation': 2.0, 'proposal': 1.5, 'amendment': 2.0,
                'draft': 1.5, 'article': 1.5, 'legal': 1.0, 'treaty': 2.0,
                'committee': 1.0, 'legislation': 2.0, 'codecision': 3.0
            },
            'factual': {
                'report': 1.5, 'data': 1.5, 'figures': 1.5, 'study': 1.5,
                'result': 1.0, 'analysis': 1.5, 'evidence': 2.0, 'research': 1.5,
                'document': 1.0, 'findings': 1.5, 'survey': 1.5
            },
            'argumentative': {
                'believe': 1.5, 'argue': 2.0, 'opinion': 1.5, 'position': 1.5,
                'support': 1.0, 'oppose': 1.5, 'against': 1.0, 'favor': 1.0,
                'disagree': 2.0, 'debate': 1.5, 'view': 1.0
            },
            'administrative': {
                'office': 1.5, 'secretary': 1.5, 'staff': 1.5, 'budget': 1.5,
                'resources': 1.0, 'schedule': 1.0, 'organize': 1.0, 'building': 1.0,
                'administration': 2.0, 'facilities': 1.5, 'management': 1.0
            }
        }

        # Calculate weighted score for each category
        scores = {category: 0.0 for category in category_keywords}

        # Check exact phrase matches first
        text_lower = text.lower()
        for category, keywords in category_keywords.items():
            for keyword, weight in keywords.items():
                if ' ' in keyword:  # Multi-word phrase
                    if keyword in text_lower:
                        scores[category] += weight * 1.5  # Bonus for phrase match
                else:
                    # Handle single word with word boundary check
                    for word in re.findall(r'\b\w+\b', text_lower):
                        if word == keyword:
                            scores[category] += weight

        # Find category with highest score
        max_score = 0
        max_category = 'procedural'  # Default

        for category, score in scores.items():
            if score > max_score:
                max_score = score
                max_category = category

        # Map category to index
        category_to_idx = {cat: i for i, cat in enumerate(classifier.content_classes)}
        return category_to_idx[max_category]

    # Create dataset with enhanced labels
    X = embeddings
    y = [assign_content_label(sentence) for sentence in sentences]

    # Convert to tensors
    X_tensor = torch.tensor(np.array(X), dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)

    # Training parameters
    epochs = 20
    batch_size = 32

    # Create optimizer
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, weight_decay=1e-5)

    # Create loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    logger.info(f"Training enhanced content classifier on {len(X_tensor)} examples")

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
            outputs = classifier(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
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

    return classifier

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
        classifier = train_enhanced_content_classifier(embeddings, sentences, model_path)

        return True
    except Exception as e:
        logger.error(f"Error preparing content classifier: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

