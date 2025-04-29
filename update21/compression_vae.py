import os
import spacy
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer, BertModel
from knowledge_base import get_or_create_knowledge_base
import logging
import traceback
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directory setup
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_data.pkl")
COMPRESSED_DATA_PATH = os.path.join(DATA_DIR, "compressed_data.pkl")
VAE_COMPRESSOR_PATH = os.path.join(DATA_DIR, "vae_compressor.pth")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load NLP tools
nlp = spacy.load("en_core_web_sm")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()

# Configuration options
USE_VAE_COMPRESSION = True  # Set to False to use traditional truncation compression
VAE_COMPRESSION_FACTOR = 0.6  # Compression ratio (0.6 means compressed to 60% of original size)


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


class EmbeddingCompressorVAE(nn.Module):
    """
    Variational Autoencoder for non-linear compression of semantic embeddings.
    Replaces the simple truncation-based compression with a learned approach.
    """

    def __init__(self, input_dim, compressed_dim, hidden_dim=None):
        super().__init__()

        # Set hidden dimensions if not provided
        if hidden_dim is None:
            hidden_dim = min(1024, input_dim * 2)

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
        )

        # VAE components
        self.fc_mu = nn.Linear(hidden_dim // 2, compressed_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, compressed_dim)

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim)
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Track dimensions
        self.input_dim = input_dim
        self.compressed_dim = compressed_dim
        self.hidden_dim = hidden_dim

    def _init_weights(self, module):
        """Initialize weights for better training"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=0.7)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(self, x):
        """Encode input to latent parameters"""
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """VAE reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent vector to reconstruction"""
        return self.decoder(z)

    def forward(self, x):
        """Full forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

    def compress(self, x):
        """Compress embedding for transmission (deterministic) with dimension checks"""
        # Convert to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32).to(device)

        # Check and fix dimensions
        current_dim = x.shape[1] if len(x.shape) > 1 else x.shape[0]
        if current_dim != self.input_dim:
            # Need to reshape
            x = adapt_dimensions(x, self.input_dim)

        # Ensure proper shape for encoding
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        # For inference, just use the mean vector
        with torch.no_grad():
            try:
                mu, _ = self.encode(x)
                return mu
            except Exception as e:
                logger.error(f"Error in VAE compression: {e}, input shape: {x.shape}")
                # Return input as fallback (with dimension reduction if needed)
                if x.shape[1] > self.compressed_dim:
                    return x[:, :self.compressed_dim]
                else:
                    return x

    def decompress(self, z):
        """Decompress latent vector back to embedding space"""
        with torch.no_grad():
            return self.decode(z)


class EnhancedVAECompressor(nn.Module):
    """
    Enhanced VAE compressor with residual connections and attention mechanisms
    for better semantic preservation during compression.
    """

    def __init__(self, input_dim, compressed_dim, hidden_dim=None):
        super().__init__()

        # Set hidden dimensions if not provided
        if hidden_dim is None:
            hidden_dim = min(1024, input_dim * 2)

        # Store dimensions
        self.input_dim = input_dim
        self.compressed_dim = compressed_dim
        self.hidden_dim = hidden_dim

        # Encoder with residual connections
        self.encoder_block1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )

        self.encoder_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )

        self.encoder_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Sigmoid()
        )

        # VAE components
        self.fc_mu = nn.Linear(hidden_dim, compressed_dim)
        self.fc_logvar = nn.Linear(hidden_dim, compressed_dim)

        # Decoder with residual connections
        self.decoder_block1 = nn.Sequential(
            nn.Linear(compressed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )

        self.decoder_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )

        self.decoder_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Sigmoid()
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights for better training"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=0.7)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(self, x):
        """Encode input to latent parameters with attention and residuals"""
        # First encoder block
        h1 = self.encoder_block1(x)

        # Second encoder block with residual connection
        h2 = self.encoder_block2(h1)
        h2 = h1 + h2  # Residual connection

        # Apply attention
        attention = self.encoder_attention(h2)
        h_attended = h2 * attention

        # Get latent parameters
        mu = self.fc_mu(h_attended)
        logvar = self.fc_logvar(h_attended)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """VAE reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent vector with attention and residuals"""
        # First decoder block
        h1 = self.decoder_block1(z)

        # Second decoder block with residual connection
        h2 = self.decoder_block2(h1)
        h2 = h1 + h2  # Residual connection

        # Apply attention
        attention = self.decoder_attention(h2)
        h_attended = h2 * attention

        # Output layer
        reconstructed = self.output_layer(h_attended)

        return reconstructed

    def forward(self, x):
        """Full forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

    def compress(self, x):
        """Compress embedding for transmission (deterministic)"""
        # Convert to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32).to(device)

        # Ensure proper shape
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        # For inference, just use the mean vector
        with torch.no_grad():
            mu, _ = self.encode(x)
            return mu

    def decompress(self, z):
        """Decompress latent vector back to embedding space"""
        with torch.no_grad():
            return self.decode(z)

# In compression_vae.py, add after EmbeddingCompressorVAE class
class EnhancedEmbeddingVAE(nn.Module):
    """
    Advanced VAE with attention mechanisms and residual connections
    for better semantic embedding compression.
    """

    def __init__(self, input_dim, compressed_dim, hidden_dim=None):
        super().__init__()

        # Set hidden dimension if not provided
        if hidden_dim is None:
            hidden_dim = min(1024, input_dim * 2)

        # Store dimensions
        self.input_dim = input_dim
        self.compressed_dim = compressed_dim
        self.hidden_dim = hidden_dim

        # Residual block for encoder
        self.encoder_res1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Encoder attention
        self.encoder_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 8),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 8, hidden_dim),
            nn.Sigmoid()
        )

        # Encoder final layers
        self.encoder_final = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
        )

        # VAE components
        self.fc_mu = nn.Linear(hidden_dim // 2, compressed_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, compressed_dim)

        # Decoder first layers
        self.decoder_first = nn.Sequential(
            nn.Linear(compressed_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
        )

        # Residual block for decoder
        self.decoder_res1 = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Decoder attention
        self.decoder_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 8),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 8, hidden_dim),
            nn.Sigmoid()
        )

        # Decoder final layers
        self.decoder_final = nn.Linear(hidden_dim, input_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights for better training"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=0.7)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(self, x):
        """Encode with residual connections and attention"""
        # Ensure batch dimension
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        # First residual block
        res1_out = self.encoder_res1(x)
        x = x.expand(-1, self.hidden_dim) if x.shape[1] != self.hidden_dim else x
        x = x + res1_out  # Residual connection

        # Apply attention
        attention = self.encoder_attention(x)
        x = x * attention  # Attention modulation

        # Final encoding layers
        x = self.encoder_final(x)

        # Get latent parameters
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """VAE reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode with residual connections and attention"""
        # First decoder layers
        x = self.decoder_first(z)

        # Residual block
        res1_out = self.decoder_res1(x)
        x = x.expand(-1, self.hidden_dim) if x.shape[1] != self.hidden_dim else x
        x = res1_out  # No residual here since dimensions differ

        # Apply attention
        attention = self.decoder_attention(x)
        x = x * attention  # Attention modulation

        # Final decoding
        x = self.decoder_final(x)

        return x

    def forward(self, x):
        """Full forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

    def compress(self, x):
        """Compress embedding for transmission (deterministic)"""
        # For inference, just use the mean vector
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu

    def decompress(self, z):
        """Decompress latent vector back to embedding space"""
        with torch.no_grad():
            return self.decode(z)
def get_semantic_embedding(sentence, max_length=512, use_kb=True):
    """
    Generate BERT embedding for a sentence with knowledge base enhancement.
    """
    # Skip empty sentences
    if not sentence or not sentence.strip():
        logger.warning("Empty sentence provided to embedding function")
        return np.zeros(768)  # Return zero vector

    # Tokenize sentence
    tokenized = bert_tokenizer.tokenize(sentence)
    embeddings = []

    # Split tokenized input into chunks of max_length if needed
    for i in range(0, len(tokenized), max_length):
        chunk = bert_tokenizer.convert_tokens_to_string(tokenized[i:i + max_length])

        # Skip empty chunks
        if not chunk.strip():
            continue

        # Get embedding for chunk
        inputs = bert_tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(
            device)

        with torch.no_grad():
            outputs = bert_model(**inputs)
            # Use mean of token embeddings as chunk embedding
            chunk_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
            embeddings.append(chunk_embedding)

    # Calculate mean of chunk embeddings
    if embeddings:
        base_embedding = np.mean(embeddings, axis=0)
    else:
        base_embedding = np.zeros(768)  # Fallback for empty input

    # Apply knowledge base enhancement if enabled
    if use_kb:
        try:
            kb = get_or_create_knowledge_base()
            enhanced_embedding = kb.enhance_embedding(base_embedding, sentence)
            return enhanced_embedding
        except Exception as e:
            logger.warning(f"Knowledge base enhancement failed: {e}, using base embedding")
            return base_embedding

    return base_embedding

def compress_embeddings(embedding, compression_factor=0.6):
    """
    Simple compression by truncation - takes the first N elements of the embedding.
    This is much simpler than PCA and works for single vectors.
    """
    # Calculate number of elements to keep
    n_elements = max(int(len(embedding) * compression_factor), 10)

    # Truncate embedding (keep only first n_elements)
    return embedding[:n_elements]


def train_vae_compressor(embeddings, compressed_dim, epochs=30, batch_size=64, learning_rate=1e-3):
    """
    Train the VAE compressor on a set of embeddings.

    Args:
        embeddings: Numpy array of embeddings
        compressed_dim: Target dimension for compression
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate

    Returns:
        Trained EmbeddingCompressorVAE model
    """
    # Convert to torch tensor
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(embeddings_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create VAE model
    input_dim = embeddings.shape[1]
    vae = EmbeddingCompressorVAE(input_dim, compressed_dim).to(device)

    # Create optimizer
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    # Training loop
    vae.train()
    logger.info(f"Training VAE compressor: {input_dim} → {compressed_dim} dimensions")

    best_loss = float('inf')
    patience_counter = 0
    patience = 5  # Early stopping patience

    for epoch in range(epochs):
        total_loss = 0
        recon_loss_sum = 0
        kl_loss_sum = 0
        num_batches = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            # Get batch
            x = batch[0].to(device)

            # Forward pass
            reconstructed, mu, logvar = vae(x)

            # Calculate loss
            recon_loss = F.mse_loss(reconstructed, x)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

            # Total loss (beta-VAE formulation with beta=0.01 to prioritize reconstruction)
            loss = recon_loss + 0.01 * kl_loss

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            num_batches += 1

        # Calculate average loss
        avg_loss = total_loss / num_batches
        avg_recon_loss = recon_loss_sum / num_batches
        avg_kl_loss = kl_loss_sum / num_batches

        logger.info(f"Epoch {epoch + 1}: Loss = {avg_loss:.6f}, Recon = {avg_recon_loss:.6f}, KL = {avg_kl_loss:.6f}")

        # Save best model (early stopping)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': vae.state_dict(),
                'input_dim': input_dim,
                'compressed_dim': compressed_dim,
                'hidden_dim': vae.hidden_dim,
                'loss': best_loss
            }, VAE_COMPRESSOR_PATH)
            logger.info(f"Saved improved model with loss {best_loss:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break

    # Load best model
    checkpoint = torch.load(VAE_COMPRESSOR_PATH)
    vae.load_state_dict(checkpoint['model_state_dict'])

    logger.info(f"VAE Compressor training complete. Best loss: {best_loss:.6f}")
    return vae


def compress_embeddings_vae(embedding, vae_compressor):
    """
    Compress embedding using trained VAE (replaces simple truncation).

    Args:
        embedding: Numpy array embedding to compress
        vae_compressor: Trained EmbeddingCompressorVAE model

    Returns:
        Compressed embedding as numpy array
    """
    # Convert to tensor
    if isinstance(embedding, np.ndarray):
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).to(device)
    else:
        embedding_tensor = embedding.to(device)

    # Handle batch vs single embedding
    if len(embedding_tensor.shape) == 1:
        embedding_tensor = embedding_tensor.unsqueeze(0)

    # Compress using VAE
    with torch.no_grad():
        compressed = vae_compressor.compress(embedding_tensor)

    # Back to numpy and handle batch vs single
    compressed_np = compressed.cpu().numpy()
    if len(embedding.shape) == 1:
        compressed_np = compressed_np.squeeze(0)

    return compressed_np


def decompress_vae_embedding(compressed_embedding):
    """
    Decompress an embedding that was compressed with the VAE.

    Args:
        compressed_embedding: VAE-compressed embedding

    Returns:
        Decompressed embedding in original space
    """
    # Load VAE compressor
    try:
        checkpoint = torch.load(VAE_COMPRESSOR_PATH, map_location=device)
        vae = EmbeddingCompressorVAE(
            checkpoint['input_dim'],
            checkpoint['compressed_dim'],
            checkpoint.get('hidden_dim')
        ).to(device)
        vae.load_state_dict(checkpoint['model_state_dict'])
        vae.eval()

        # Convert to tensor
        if isinstance(compressed_embedding, np.ndarray):
            compressed_tensor = torch.tensor(compressed_embedding, dtype=torch.float32).to(device)
        else:
            compressed_tensor = compressed_embedding.to(device)

        # Handle batch vs single embedding
        if len(compressed_tensor.shape) == 1:
            compressed_tensor = compressed_tensor.unsqueeze(0)

        # Decompress
        with torch.no_grad():
            decompressed = vae.decompress(compressed_tensor)

        # Back to numpy and handle batch vs single
        decompressed_np = decompressed.cpu().numpy()
        if len(compressed_embedding.shape) == 1:
            decompressed_np = decompressed_np.squeeze(0)

        return decompressed_np
    except Exception as e:
        logger.error(f"Error decompressing with VAE: {e}")
        # Return the input as fallback
        return compressed_embedding


def load_or_train_vae_compressor(compression_factor=VAE_COMPRESSION_FACTOR, embedding_dim=None, use_enhanced=True):
    """
    Load existing VAE compressor or train a new one with proper dimension handling.

    Args:
        compression_factor: Factor to determine compressed dimension size
        embedding_dim: Optional explicit dimension for the embedding
        use_enhanced: Whether to use the enhanced VAE architecture

    Returns:
        Trained VAE compressor model
    """
    # Check if VAE model exists
    if os.path.exists(VAE_COMPRESSOR_PATH):
        logger.info(f"Loading existing VAE compressor from {VAE_COMPRESSOR_PATH}")
        try:
            checkpoint = torch.load(VAE_COMPRESSOR_PATH, map_location=device)

            # Check if the checkpoint contains necessary information
            if not isinstance(checkpoint, dict):
                logger.warning(f"Invalid checkpoint format: {type(checkpoint)}")
                # Train new model instead
                return train_new_compressor(compression_factor, embedding_dim)

            # Get dimensions from checkpoint
            input_dim = checkpoint.get('input_dim')
            compressed_dim = checkpoint.get('compressed_dim')
            hidden_dim = checkpoint.get('hidden_dim')

            # Validate dimensions
            if not input_dim or not compressed_dim:
                logger.warning(f"Missing dimensions in checkpoint")
                # Train new model instead
                return train_new_compressor(compression_factor, embedding_dim)

            # Check if dimensions are compatible with embedding_dim if provided
            if embedding_dim and input_dim != embedding_dim:
                logger.warning(
                    f"Dimension mismatch: checkpoint input_dim={input_dim}, requested embedding_dim={embedding_dim}")
                logger.warning(f"Will train a new model with the correct dimensions")
                # Train new model with correct dimensions
                return train_new_compressor(compression_factor, embedding_dim)

            # Check if the checkpoint is for enhanced or standard VAE
            is_enhanced = checkpoint.get('is_enhanced', False)

            # Create appropriate model
            if is_enhanced and use_enhanced:
                vae = EnhancedEmbeddingVAE(
                    input_dim=input_dim,
                    compressed_dim=compressed_dim,
                    hidden_dim=hidden_dim
                ).to(device)
            else:
                vae = EmbeddingCompressorVAE(
                    input_dim=input_dim,
                    compressed_dim=compressed_dim,
                    hidden_dim=hidden_dim
                ).to(device)

            # Load state dict with error handling
            if 'model_state_dict' not in checkpoint:
                logger.warning("Missing model_state_dict in checkpoint")
                return train_new_compressor(compression_factor, embedding_dim)

            try:
                vae.load_state_dict(checkpoint['model_state_dict'])
            except Exception as e:
                logger.warning(f"Error loading state dict: {e}")
                return train_new_compressor(compression_factor, embedding_dim)

            vae.eval()
            logger.info(f"VAE compressor loaded: {input_dim} → {compressed_dim} dimensions")

            # Indicate if architecture matches requested type
            if is_enhanced != use_enhanced:
                logger.info(
                    f"Note: Loaded {'enhanced' if is_enhanced else 'standard'} model, but requested {'enhanced' if use_enhanced else 'standard'}")

            # Register the dimensions in a global variable for other components to access
            global VAE_DIMENSIONS
            VAE_DIMENSIONS = {
                'input_dim': vae.input_dim,
                'compressed_dim': vae.compressed_dim
            }

            # Create a dimensions file that other components can read
            try:
                with open(os.path.join(DATA_DIR, 'vae_dimensions.json'), 'w') as f:
                    json.dump(VAE_DIMENSIONS, f)
                logger.info(f"Saved VAE dimensions to file: input={vae.input_dim}, compressed={vae.compressed_dim}")
            except Exception as e:
                logger.warning(f"Could not save VAE dimensions: {e}")

            return vae

        except Exception as e:
            logger.error(f"Error loading VAE compressor: {e}")
            logger.info("Will train a new model")
            return train_new_compressor(compression_factor, embedding_dim)

    # If no model exists or loading failed, train a new one
    return train_new_compressor(compression_factor, embedding_dim)


def train_new_compressor(compression_factor, embedding_dim=None):
    """Helper function to train a new compressor with proper error handling"""
    try:
        # Check if we have processed data first
        if os.path.exists(PROCESSED_DATA_PATH):
            logger.info("Loading processed data for VAE training")
            with open(PROCESSED_DATA_PATH, "rb") as f:
                processed_data = pickle.load(f)

            # Generate embeddings for VAE training
            logger.info("Generating embeddings for VAE training")
            embeddings = []
            for item in tqdm(processed_data[:1000], desc="Generating embeddings"):  # Limit to 1000 for speed
                if isinstance(item, str):
                    # It's a sentence, generate embedding
                    embedding = get_semantic_embedding(item)
                    embeddings.append(embedding)
                elif isinstance(item, dict) and 'embedding' in item:
                    # It's an embedding dictionary
                    embeddings.append(item['embedding'])
                elif isinstance(item, np.ndarray):
                    # It's already an embedding
                    embeddings.append(item)

            if not embeddings:
                logger.error("No embeddings could be generated from processed data")
                return None

            embeddings = np.array(embeddings)

            # Use provided embedding_dim or detect from data
            if embedding_dim is None:
                if len(embeddings) > 0:
                    embedding_dim = embeddings[0].shape[0]
                    logger.info(f"Detected embedding dimension: {embedding_dim}")
                else:
                    embedding_dim = 768  # Default to BERT dimension
                    logger.warning(f"No embeddings available, using default dimension: {embedding_dim}")

            # Determine target compressed dimension
            input_dim = embedding_dim
            compressed_dim = max(int(input_dim * compression_factor), 50)
            logger.info(f"Training VAE compressor: {input_dim} → {compressed_dim}")

            # Train VAE compressor
            vae = train_vae_compressor(embeddings, compressed_dim)

            # Register the dimensions in a global variable for other components to access
            global VAE_DIMENSIONS
            VAE_DIMENSIONS = {
                'input_dim': vae.input_dim,
                'compressed_dim': vae.compressed_dim
            }

            # Create a dimensions file that other components can read
            try:
                with open(os.path.join(DATA_DIR, 'vae_dimensions.json'), 'w') as f:
                    json.dump(VAE_DIMENSIONS, f)
                logger.info(f"Saved VAE dimensions to file: input={vae.input_dim}, compressed={vae.compressed_dim}")
            except Exception as e:
                logger.warning(f"Could not save VAE dimensions: {e}")

            return vae
        else:
            logger.error(f"No processed data found at {PROCESSED_DATA_PATH}")
            return None

    except Exception as e:
        logger.error(f"Error training VAE compressor: {e}")
        logger.error(traceback.format_exc())
        return None

def preprocess_europarl_data(file_path="../manual_data/english/combined.en",
                             max_samples=20000,
                             min_words=5,
                             max_words=50,
                             use_vae_compression=USE_VAE_COMPRESSION):
    """
    Enhanced preprocessing function that uses VAE for compression when enabled.

    Args:
        file_path: Path to Europarl dataset
        max_samples: Maximum number of samples to process
        min_words: Minimum word count for sentences
        max_words: Maximum word count for sentences
        use_vae_compression: Whether to use VAE compression or simple truncation

    Returns:
        Tuple of (num_sentences, num_compressed)
    """
    logger.info(f"Loading data from: {file_path}")

    # Validate file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        logger.info("Please check the path and ensure the file exists")
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    # Read file with proper encoding
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            logger.info("Reading file contents...")
            data = file.readlines()
        logger.info(f"Successfully read {len(data)} lines from file")
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise

    # Process sentences
    logger.info("Tokenizing text into sentences...")
    all_sentences = []
    for line in tqdm(data, desc="Processing lines"):
        line = line.strip()
        if not line:
            continue

        # Split into sentences
        for sentence in sent_tokenize(line):
            # Clean and filter by length
            sentence = sentence.strip()
            word_count = len(sentence.split())

            if min_words <= word_count <= max_words:
                all_sentences.append(sentence)

            # Stop if we have enough samples
            if len(all_sentences) >= max_samples:
                break

        if len(all_sentences) >= max_samples:
            break

    logger.info(f"Extracted {len(all_sentences)} valid sentences")

    # Save processed sentences
    with open(PROCESSED_DATA_PATH, "wb") as f:
        pickle.dump(all_sentences, f)
    logger.info(f"Saved processed sentences to {PROCESSED_DATA_PATH}")

    # Initialize VAE compressor if using VAE compression
    vae_compressor = None
    if use_vae_compression:
        logger.info("Using VAE for advanced compression")
        # First generate regular embeddings for all sentences
        logger.info("Generating initial embeddings for VAE training")
        raw_embeddings = []
        for sentence in tqdm(all_sentences, desc="Generating embeddings"):
            embedding = get_semantic_embedding(sentence)
            raw_embeddings.append(embedding)

        # Now train or load the VAE compressor
        input_dim = raw_embeddings[0].shape[0]
        compressed_dim = max(int(input_dim * VAE_COMPRESSION_FACTOR), 50)

        # Check if we already have a trained compressor
        if os.path.exists(VAE_COMPRESSOR_PATH):
            try:
                # Load existing VAE compressor
                checkpoint = torch.load(VAE_COMPRESSOR_PATH, map_location=device)
                vae_compressor = EmbeddingCompressorVAE(
                    checkpoint['input_dim'],
                    checkpoint['compressed_dim'],
                    checkpoint.get('hidden_dim')
                ).to(device)
                vae_compressor.load_state_dict(checkpoint['model_state_dict'])
                vae_compressor.eval()
                logger.info(
                    f"Loaded existing VAE compressor: {checkpoint['input_dim']} → {checkpoint['compressed_dim']}")
            except Exception as e:
                logger.warning(f"Error loading VAE compressor: {e}")
                vae_compressor = None

        # If loading failed, train a new one
        if vae_compressor is None:
            logger.info("Training new VAE compressor")
            vae_compressor = train_vae_compressor(
                np.array(raw_embeddings), compressed_dim, epochs=20)

        if vae_compressor is None:
            logger.warning("VAE compressor initialization failed, falling back to truncation")
            use_vae_compression = False

    # Generate and compress embeddings
    logger.info("Generating compressed embeddings...")
    compressed_data = []

    for i, sentence in enumerate(tqdm(all_sentences, desc="Processing embeddings")):
        try:
            # Generate BERT embedding or reuse from VAE training
            if use_vae_compression and 'raw_embeddings' in locals() and i < len(raw_embeddings):
                embedding = raw_embeddings[i]
            else:
                embedding = get_semantic_embedding(sentence)

            # Compress embedding (either with VAE or truncation)
            if use_vae_compression and vae_compressor is not None:
                compressed_embedding = compress_embeddings_vae(embedding, vae_compressor)
                compression_method = "vae"
            else:
                compressed_embedding = compress_embeddings(embedding)
                compression_method = "truncation"

            # Store the compressed embedding paired with sentence
            compressed_data.append({
                "sentence": sentence,
                "embedding": compressed_embedding,
                "compression_method": compression_method
            })

        except Exception as e:
            logger.warning(f"Error processing sentence: {e}")
            logger.warning(traceback.format_exc())

    logger.info(f"Generated {len(compressed_data)} compressed embeddings")

    # Save compressed data
    with open(COMPRESSED_DATA_PATH, "wb") as f:
        pickle.dump(compressed_data, f)
    logger.info(f"Saved {len(compressed_data)} compressed embeddings to {COMPRESSED_DATA_PATH}")

    # Save compression configuration for future reference
    compression_config = {
        "use_vae_compression": use_vae_compression,
        "compression_factor": VAE_COMPRESSION_FACTOR if use_vae_compression else 0.6,
        "input_dim": input_dim if use_vae_compression else None,
        "compressed_dim": compressed_dim if use_vae_compression else None
    }

    with open(os.path.join(DATA_DIR, "compression_config.json"), "w") as f:
        json.dump(compression_config, f, indent=2)

    return len(all_sentences), len(compressed_data)


if __name__ == "__main__":
    try:
        # Use the exact path provided for Europarl dataset
        europarl_path = "../manual_data/english/combined.en"
        logger.info(f"Starting preprocessing with file: {europarl_path}")

        # Run preprocessing with VAE compression
        num_sentences, num_compressed = preprocess_europarl_data(
            file_path=europarl_path,
            max_samples=10000,  # Process fewer samples for speed
            min_words=5,
            max_words=40,
            use_vae_compression=USE_VAE_COMPRESSION
        )

        logger.info("=== Preprocessing Complete ===")
        logger.info(f"Processed sentences: {num_sentences}")
        logger.info(f"Compressed embeddings: {num_compressed}")
        if USE_VAE_COMPRESSION:
            logger.info("Used VAE-based compression")
        else:
            logger.info("Used truncation-based compression")

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        logger.error(traceback.format_exc())
        logger.info("Please check file paths and permissions")