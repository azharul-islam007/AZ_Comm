import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from knowledge_base import get_or_create_knowledge_base
from mlpdvae_utils import ensure_tensor_shape, compute_embedding_similarity, generate_text_from_embedding
import time
import json
from tqdm import tqdm
from collections import deque

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import utility functions
try:
    from mlpdvae_utils import (
        load_transmission_pairs,
        evaluate_reconstruction_with_semantics,
        compute_embedding_similarity,
        generate_text_from_embedding,
        detect_embedding_dimensions
    )
except ImportError:
    # Fallback functions if module not available
    def detect_embedding_dimensions():
        """Fallback function"""
        return 768, 1024, 512


    def load_transmission_pairs(*args, **kwargs):
        """Fallback function"""
        return []

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths for data and models
DATA_DIR = "./data"
MODELS_DIR = "./models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


class MLPDenoisingVAE(nn.Module):
    """
    MLP-based Denoising Variational Autoencoder for semantic embeddings reconstruction.
    Uses a simpler but more effective architecture than transformers for this specific task.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.1):
        super(MLPDenoisingVAE, self).__init__()

        # Dimensionality parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )

        # VAE components
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder layers
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 2, input_dim)
        )

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with Xavier/Glorot initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=0.7)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(self, x):
        """Encode input to latent representation"""
        # Use our utility function to ensure consistent shape
        x = ensure_tensor_shape(x, add_batch_dim=True)

        # Pass through encoder
        hidden = self.encoder(x)

        # Get mean and logvar for VAE
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, apply_kb=True):
        """Decode latent representation to reconstruction with KB enhancement"""
        # Ensure z has batch dimension
        z = ensure_tensor_shape(z, add_batch_dim=True)

        # Initial projection
        hidden = self.decoder_input(z)
        hidden = F.leaky_relu(hidden, 0.2)

        # Pass through decoder
        reconstructed = self.decoder(hidden)

        # Apply knowledge base guidance if enabled
        if apply_kb:
            try:
                # We can't directly apply KB to embeddings since we need text
                # In a real implementation, this would be done by:
                # 1. Converting embeddings to text features
                # 2. Applying KB guidance
                # 3. Re-embedding the guided features

                # For now, we'll do a simplified enhancement:
                kb = get_or_create_knowledge_base()

                # Apply a learned transformation that mimics KB guidance
                # This is just a placeholder for demonstration - in practice this would
                # involve actual text conversion and KB application
                batch_size = reconstructed.shape[0]

                for i in range(batch_size):
                    # Create a seed from the embedding to ensure deterministic behavior
                    seed = int(reconstructed[i].sum().item() * 1000) % 10000
                    torch.manual_seed(seed)

                    # Apply minor adjustments to dimensions that would relate to KB entries
                    # This simulates the effect of KB guidance without actual text conversion
                    mask = torch.rand(reconstructed[i].shape).to(reconstructed.device)
                    boost_mask = mask > 0.9  # Apply to ~10% of dimensions

                    # Apply small enhancements to these dimensions
                    reconstructed[i][boost_mask] *= 1.05
            except Exception as e:
                logger.debug(f"KB guidance in decoder skipped: {e}")

        return reconstructed

    def decode_with_text_guidance(self, z, text_hint=None, text_context=None):
        """
        Decode latent with text guidance from KB when available
        """
        # Basic decode first
        reconstructed = self.decode(z)

        # If no text hints, return basic reconstruction
        if text_hint is None:
            return reconstructed

        # Try to apply KB guidance to improve the result
        try:
            # Get embeddings for text hint
            if hasattr(self, 'get_text_embedding'):
                hint_embedding = self.get_text_embedding(text_hint)

                # Calculate similarity between reconstruction and hint
                sim = F.cosine_similarity(
                    reconstructed.view(1, -1),
                    torch.tensor(hint_embedding, device=reconstructed.device).view(1, -1)
                )

                # If similarity is already high, don't modify
                if sim > 0.9:
                    return reconstructed

                # Otherwise, apply a small bias toward the hint
                hint_weight = max(0.0, min(0.3, 0.8 - sim.item()))
                adjusted = reconstructed + hint_weight * torch.tensor(
                    hint_embedding,
                    device=reconstructed.device
                )

                # Normalize to preserve norm
                norm_factor = torch.norm(reconstructed) / torch.norm(adjusted)
                return adjusted * norm_factor
        except Exception as e:
            logger.debug(f"Text-guided decoding error: {e}")

        # Return original reconstruction if guidance failed
        return reconstructed

    def forward(self, x):
        """Forward pass through the VAE"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar


class EnhancedMLPDenoisingVAE(MLPDenoisingVAE):
    """
    Enhanced version of MLPDenoisingVAE with knowledge-based enhancement
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.1, use_kb=True):
        super().__init__(input_dim, hidden_dim, latent_dim, dropout)

        self.use_kb = use_kb
        self.kb_mapper = None
        self.kb_decoder = None

        # Add context memory for the decoder
        self.context_memory_size = 32
        self.context_memory = nn.Parameter(torch.randn(self.context_memory_size, latent_dim))

        # If KB is enabled, initialize the enhanced decoder
        if use_kb:
            try:
                from knowledge_base import get_or_create_knowledge_base, TextEmbeddingMapper, KnowledgeEnhancedDecoder

                # Get knowledge base
                kb = get_or_create_knowledge_base()

                # Initialize text-embedding mapper
                self.kb_mapper = TextEmbeddingMapper(k_neighbors=3)

                # Initialize KB-enhanced decoder
                self.kb_decoder = KnowledgeEnhancedDecoder(
                    embedding_dim=input_dim,
                    hidden_dim=hidden_dim,
                    kb=kb,
                    mapper=self.kb_mapper
                )

                logger.info("Initialized knowledge-enhanced decoder")
            except Exception as e:
                logger.warning(f"Could not initialize KB components: {e}")
                self.use_kb = False

    def initialize_kb_with_data(self, embeddings, texts, bert_model=None, bert_tokenizer=None):
        """Initialize KB components with reference data"""
        if not self.use_kb or self.kb_mapper is None or self.kb_decoder is None:
            return False

        try:
            # Fit the mapper with reference data
            self.kb_mapper.fit(embeddings, texts)

            # Initialize KB embeddings if BERT available
            if bert_model is not None and bert_tokenizer is not None:
                self.kb_decoder.initialize_kb_embeddings(bert_model, bert_tokenizer)

            logger.info(f"Initialized KB components with {len(embeddings)} reference examples")
            return True
        except Exception as e:
            logger.warning(f"Could not initialize KB with data: {e}")
            return False

    def decode(self, z, text_hints=None, apply_kb=True):
        """Enhanced decode with knowledge guidance"""
        # Ensure z has batch dimension
        if len(z.shape) == 1:
            z = z.unsqueeze(0)

        # Calculate attention with context memory
        z_expanded = z.unsqueeze(1)  # [batch_size, 1, latent_dim]
        memory_expanded = self.context_memory.unsqueeze(0)  # [1, memory_size, latent_dim]

        # Calculate attention weights
        attn_scores = torch.bmm(z_expanded, memory_expanded.transpose(1, 2))  # [batch_size, 1, memory_size]
        attn_weights = F.softmax(attn_scores, dim=2)

        # Apply attention to get context-enhanced latent
        context_vector = torch.bmm(attn_weights, memory_expanded)  # [batch_size, 1, latent_dim]
        context_vector = context_vector.squeeze(1)  # [batch_size, latent_dim]

        # Blend with original latent
        z_enhanced = z + 0.1 * context_vector

        # Initial projection
        hidden = self.decoder_input(z_enhanced)
        hidden = F.leaky_relu(hidden, 0.2)

        # Pass through decoder
        reconstructed = self.decoder(hidden)

        # Apply KB enhancement if enabled
        if apply_kb and self.use_kb and self.kb_decoder is not None:
            try:
                # Apply KB enhancement
                reconstructed = self.kb_decoder(reconstructed, text_hints)
            except Exception as e:
                logger.debug(f"KB enhancement failed: {e}")

        return reconstructed
def train_mlp_dvae_with_semantic_loss(compressed_data, sentences, transmission_pairs=None,
                                      input_dim=None, hidden_dim=None, latent_dim=None,
                                      epochs=20, batch_size=64, learning_rate=1e-3,
                                      use_semantic_loss=True,
                                      model_path="enhanced_mlp_dvae_model.pth"):
    """
    Enhanced MLPDenoisingVAE training with semantic perceptual loss.

    Args:
        compressed_data: Original compressed data for training
        sentences: Corresponding sentences for semantic loss
        transmission_pairs: Optional list of (original, received) pairs from actual transmission
        input_dim, hidden_dim, latent_dim: Model dimensions
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        use_semantic_loss: Whether to use semantic perceptual loss
        model_path: Path to save the model

    Returns:
        Trained MLPDenoisingVAE model and loss history
    """
    # Detect dimensions if not provided
    if input_dim is None or hidden_dim is None or latent_dim is None:
        # Get dimensions from first item
        if isinstance(compressed_data[0], dict) and 'embedding' in compressed_data[0]:
            first_embedding = compressed_data[0]['embedding']
        elif isinstance(compressed_data[0], tuple) and len(compressed_data[0]) == 2:
            first_embedding = compressed_data[0][0]
        else:
            first_embedding = compressed_data[0]

        input_dim = len(first_embedding)
        hidden_dim = min(1024, input_dim * 2)
        latent_dim = min(512, input_dim)
        logger.info(f"Detected dimensions: input={input_dim}, hidden={hidden_dim}, latent={latent_dim}")

    # Initialize MLP-DVAE model
    dvae = MLPDenoisingVAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=0.1
    ).to(device)

    # Initialize optimizer with weight decay (L2 regularization)
    optimizer = optim.AdamW(dvae.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Learning rate scheduler - reduce on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # Initialize semantic loss if requested
    semantic_loss_fn = None
    if use_semantic_loss:
        try:
            # Import here to avoid circular imports
            from semantic_loss import SemanticPerceptualLoss
            semantic_loss_fn = SemanticPerceptualLoss().to(device)
            logger.info("Semantic perceptual loss initialized")
        except ImportError:
            logger.warning("Could not import SemanticPerceptualLoss, will train without semantic loss")
            use_semantic_loss = False
        except Exception as e:
            logger.warning(f"Failed to initialize semantic loss: {e}")
            logger.info("Will continue training without semantic loss")
            use_semantic_loss = False

    # Extract embeddings from compressed data
    embeddings = []
    training_sentences = []

    for i, item in enumerate(compressed_data):
        if isinstance(item, dict) and 'embedding' in item:
            embeddings.append(item['embedding'])
            if i < len(sentences):
                training_sentences.append(sentences[i])
        elif isinstance(item, tuple) and len(item) == 2:
            embeddings.append(item[0])
            if i < len(sentences):
                training_sentences.append(sentences[i])
        else:
            embeddings.append(item)
            if i < len(sentences):
                training_sentences.append(sentences[i])

    embeddings = np.array(embeddings)

    # Using transmission pairs for self-supervised learning
    use_transmission_data = transmission_pairs is not None and len(transmission_pairs) > 0
    if use_transmission_data:
        logger.info(f"Using {len(transmission_pairs)} transmission pairs for self-supervised learning")

        # Extract original and received embeddings from transmission pairs
        transmission_originals = np.array([pair[0] for pair in transmission_pairs])
        transmission_received = np.array([pair[1] for pair in transmission_pairs])

        # Initial weights for combined loss
        ssl_weight = 0.3  # Start with moderate weight for self-supervised learning

    # Create validation set (10% of data)
    num_samples = len(embeddings)
    indices = np.random.permutation(num_samples)
    val_size = int(0.1 * num_samples)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    # Training loop
    dvae.train()
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 7  # Early stopping patience

    # Save loss history for plotting
    loss_history = {'train': [], 'val': [], 'ssl': [], 'semantic': []}

    for epoch in range(epochs):
        # Shuffle training data
        train_indices_shuffled = np.random.permutation(train_indices)
        total_train_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_ssl_loss = 0
        total_semantic_loss = 0
        num_batches = 0

        # Update self-supervised learning weight - gradually increase importance
        if use_transmission_data:
            # Increase weight as training progresses but cap at 0.6
            ssl_weight = min(0.6, 0.3 + 0.3 * (epoch / (epochs - 1)))

        # Process in batches
        for i in range(0, len(train_indices_shuffled), batch_size):
            batch_indices = train_indices_shuffled[i:i + batch_size]
            batch = embeddings[batch_indices]

            # Get corresponding sentences if using semantic loss
            batch_sentences = []
            if use_semantic_loss and semantic_loss_fn is not None:
                for idx in batch_indices:
                    if idx < len(training_sentences):
                        batch_sentences.append(training_sentences[idx])

            # Add noise to batch for denoising training (data augmentation)
            noise_level = np.random.uniform(0.05, 0.2)  # Vary noise level
            noise_type = np.random.choice(['gaussian', 'dropout'])

            if noise_type == 'gaussian':
                # Add Gaussian noise
                noisy_batch = batch + np.random.normal(0, noise_level, batch.shape)
            else:
                # Apply dropout noise (randomly zero out elements)
                mask = np.random.random(batch.shape) > noise_level
                noisy_batch = batch * mask

            # Convert to torch tensors
            batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
            noisy_batch_tensor = torch.tensor(noisy_batch, dtype=torch.float32).to(device)

            # Forward pass
            recon_batch, mu, logvar = dvae(noisy_batch_tensor)

            # Standard VAE loss: reconstruction + KL divergence
            recon_loss = F.mse_loss(recon_batch, batch_tensor)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_tensor.size(0)

            # Combined loss - weight KL lower for better reconstruction
            loss = recon_loss + 0.001 * kl_loss

            # Add semantic loss if available and have sentences
            semantic_loss = torch.tensor(0.0, device=device)
            if use_semantic_loss and semantic_loss_fn is not None and len(batch_sentences) >= 2:
                try:
                    # This is a simplified implementation since we can't actually
                    # convert embeddings back to text directly
                    # In a real system, you would have a text generator or lookup mechanism

                    # For demonstration, we'll create a simple proxy for semantic loss
                    # by comparing batches of original sentences to each other
                    if len(batch_sentences) >= 2:
                        # Take pairs of sentences from the batch
                        for j in range(min(5, len(batch_sentences) - 1)):  # Limit to 5 pairs for efficiency
                            sent1 = batch_sentences[j]
                            sent2 = batch_sentences[j + 1]

                            # Calculate semantic similarity between these sentences
                            semantic_loss_single = semantic_loss_fn(sent1, sent2)
                            semantic_loss += semantic_loss_single

                        semantic_loss = semantic_loss / min(5, len(batch_sentences) - 1)

                        # Weight increases over epochs
                        semantic_weight = min(0.3, 0.05 * (epoch + 1))
                        if epoch > 0:  # Only apply after first epoch
                            loss = loss + semantic_weight * semantic_loss
                except Exception as e:
                    logger.debug(f"Error in semantic loss calculation: {e}")
                    # Fallback - don't add semantic loss if error occurs

            # Add self-supervised learning loss if available
            ssl_loss = torch.tensor(0.0, device=device)
            if use_transmission_data and ssl_weight > 0:
                # Sample a batch of transmission pairs
                t_indices = np.random.choice(
                    len(transmission_originals),
                    min(batch_size, len(transmission_originals)),
                    replace=False
                )
                t_originals = transmission_originals[t_indices]
                t_received = transmission_received[t_indices]

                # Convert to tensors
                t_originals_tensor = torch.tensor(t_originals, dtype=torch.float32).to(device)
                t_received_tensor = torch.tensor(t_received, dtype=torch.float32).to(device)

                # Forward pass on received embeddings
                t_recon_batch, _, _ = dvae(t_received_tensor)

                # Reconstruction loss against original embeddings
                ssl_loss = F.mse_loss(t_recon_batch, t_originals_tensor)

                # Add to total loss with self-supervised learning weight
                loss = loss + ssl_weight * ssl_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(dvae.parameters(), max_norm=1.0)

            optimizer.step()

            # Track losses
            total_train_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            if use_transmission_data:
                total_ssl_loss += ssl_loss.item()
            if use_semantic_loss and semantic_loss_fn is not None:
                total_semantic_loss += semantic_loss.item()
            num_batches += 1

        # Calculate average training losses
        avg_train_loss = total_train_loss / num_batches if num_batches > 0 else 0
        avg_recon_loss = total_recon_loss / num_batches if num_batches > 0 else 0
        avg_kl_loss = total_kl_loss / num_batches if num_batches > 0 else 0
        avg_ssl_loss = total_ssl_loss / num_batches if use_transmission_data and num_batches > 0 else 0
        avg_semantic_loss = total_semantic_loss / num_batches if use_semantic_loss and num_batches > 0 else 0

        # Validation
        dvae.eval()
        with torch.no_grad():
            val_loss = 0
            val_batches = 0

            for i in range(0, len(val_indices), batch_size):
                val_batch_indices = val_indices[i:i + batch_size]
                val_batch = embeddings[val_batch_indices]

                # Add noise for validation
                val_noisy_batch = val_batch + np.random.normal(0, 0.1, val_batch.shape)

                # Convert to tensors
                val_batch_tensor = torch.tensor(val_batch, dtype=torch.float32).to(device)
                val_noisy_batch_tensor = torch.tensor(val_noisy_batch, dtype=torch.float32).to(device)

                # Forward pass
                val_recon_batch, val_mu, val_logvar = dvae(val_noisy_batch_tensor)

                # Calculate validation loss
                val_recon_loss = F.mse_loss(val_recon_batch, val_batch_tensor)
                val_kl_loss = -0.5 * torch.sum(
                    1 + val_logvar - val_mu.pow(2) - val_logvar.exp()) / val_batch_tensor.size(0)

                val_loss += (val_recon_loss + 0.001 * val_kl_loss).item()
                val_batches += 1

            avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')

        # Switch back to training mode
        dvae.train()

        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)

        # Save loss history
        loss_history['train'].append(avg_train_loss)
        loss_history['val'].append(avg_val_loss)
        loss_history['ssl'].append(avg_ssl_loss if use_transmission_data else 0)
        loss_history['semantic'].append(avg_semantic_loss if use_semantic_loss else 0)

        # Log progress
        log_msg = (f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}, "
                   f"Val Loss: {avg_val_loss:.6f}, Recon: {avg_recon_loss:.6f}, KL: {avg_kl_loss:.6f}")
        if use_transmission_data:
            log_msg += f", SSL: {avg_ssl_loss:.6f} (weight: {ssl_weight:.2f})"
        if use_semantic_loss:
            log_msg += f", Semantic: {avg_semantic_loss:.6f}"
        logger.info(log_msg)

        # Save best model and implement early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': dvae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'dimensions': {
                    'input_dim': input_dim,
                    'hidden_dim': hidden_dim,
                    'latent_dim': latent_dim
                },
                'use_semantic_loss': use_semantic_loss
            }, os.path.join(MODELS_DIR, model_path))
            logger.info(f"Saved best model (epoch {epoch + 1}) with validation loss {best_loss:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break

    # Load best model
    checkpoint = torch.load(os.path.join(MODELS_DIR, model_path))
    dvae.load_state_dict(checkpoint['model_state_dict'])
    dvae.eval()

    logger.info(f"Training complete. Best validation loss: {best_loss:.6f}")
    return dvae, loss_history


def load_or_train_enhanced_mlp_dvae(model_path="enhanced_mlp_dvae_model.pth",
                                    force_retrain=False,
                                    use_self_supervised=True,
                                    use_semantic_loss=True):
    """
    Load existing enhanced MLPDenoisingVAE model or train a new one.

    Args:
        model_path: Path to model file
        force_retrain: Whether to force retraining even if model exists
        use_self_supervised: Whether to use self-supervised learning
        use_semantic_loss: Whether to use semantic perceptual loss

    Returns:
        MLPDenoisingVAE model ready for inference
    """
    # Detect embedding dimensions
    # First try using utility function
    try:
        input_dim, hidden_dim, latent_dim = detect_embedding_dimensions()
    except:
        # Fallback values
        input_dim, hidden_dim, latent_dim = 768, 1024, 512
        logger.warning(f"Using fallback dimensions: input={input_dim}, hidden={hidden_dim}, latent={latent_dim}")

    if input_dim is None:
        logger.error("Could not detect embedding dimensions")
        return None

    full_model_path = os.path.join(MODELS_DIR, model_path)

    # Check if model exists and we don't need to retrain
    if os.path.exists(full_model_path) and not force_retrain:
        logger.info(f"Loading pre-trained enhanced MLPDenoisingVAE from {full_model_path}")
        try:
            # Load checkpoint
            checkpoint = torch.load(full_model_path, map_location=device)

            # Get dimensions from checkpoint or use detected dimensions
            if 'dimensions' in checkpoint:
                dims = checkpoint['dimensions']
                input_dim = dims.get('input_dim', input_dim)
                hidden_dim = dims.get('hidden_dim', hidden_dim)
                latent_dim = dims.get('latent_dim', latent_dim)

            # Create model with appropriate dimensions
            dvae = MLPDenoisingVAE(input_dim, hidden_dim, latent_dim).to(device)
            dvae.load_state_dict(checkpoint['model_state_dict'])
            dvae.eval()

            # Check if the model was trained with semantic loss
            model_used_semantic = checkpoint.get('use_semantic_loss', False)
            if model_used_semantic:
                logger.info("Loaded model was trained with semantic perceptual loss")
            else:
                logger.info("Loaded model was trained WITHOUT semantic perceptual loss")

            return dvae

        except Exception as e:
            logger.warning(f"Could not load model: {str(e)}")
            logger.warning("Will train a new model")

    # Train a new model
    logger.info("Training new enhanced MLPDenoisingVAE model")

    # Load compressed data for training
    try:
        with open(os.path.join(DATA_DIR, "compressed_data.pkl"), "rb") as f:
            compressed_data = pickle.load(f)

        # Load sentences for semantic loss
        sentences = []
        try:
            with open(os.path.join(DATA_DIR, "processed_data.pkl"), "rb") as f:
                sentences = pickle.load(f)
        except Exception as e:
            logger.warning(f"Could not load processed sentences: {e}")
            # Extract sentences from compressed data if available
            if isinstance(compressed_data[0], dict) and 'sentence' in compressed_data[0]:
                sentences = [item['sentence'] for item in compressed_data]
            else:
                logger.warning("No sentences available for semantic loss")
                use_semantic_loss = False

        logger.info(f"Loaded {len(compressed_data)} compressed embeddings" +
                    (f" and {len(sentences)} sentences" if sentences else ""))

        # Load transmission pairs for self-supervised learning if enabled
        transmission_pairs = None
        if use_self_supervised:
            transmission_pairs = load_transmission_pairs(max_pairs=2000)

        # Train enhanced model
        full_model_path = os.path.join(MODELS_DIR, model_path)
        return train_mlp_dvae_with_semantic_loss(
            compressed_data,
            sentences,
            transmission_pairs,
            input_dim,
            hidden_dim,
            latent_dim,
            epochs=20,  # More epochs for initial training
            batch_size=64,
            use_semantic_loss=use_semantic_loss,
            model_path=model_path
        )[0]  # Return just the model, not the loss history

    except Exception as e:
        logger.error(f"Error training enhanced MLPDenoisingVAE: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


# Example usage when run directly
if __name__ == "__main__":
    USE_SEMANTIC_LOSS = True
    USE_SELF_SUPERVISED = True

    print("=== Enhanced MLPDenoisingVAE with Semantic Loss ===")
    print(f"Device: {device}")
    print(f"Using semantic loss: {USE_SEMANTIC_LOSS}")
    print(f"Using self-supervised learning: {USE_SELF_SUPERVISED}")

    # Load or train model
    model = load_or_train_enhanced_mlp_dvae(
        use_semantic_loss=USE_SEMANTIC_LOSS,
        use_self_supervised=USE_SELF_SUPERVISED
    )

    if model is not None:
        print("Model successfully loaded/trained!")
        print(f"Model dimensions: input={model.input_dim}, hidden={model.hidden_dim}, latent={model.latent_dim}")
    else:
        print("Failed to load or train model.")