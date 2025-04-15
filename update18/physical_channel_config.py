# Configuration file for physical channel settings in semantic communication
# Enhanced with additional parameters for adaptive modulation & coding

# General physical channel settings
ENABLE_PHYSICAL_CHANNEL = True  # Set to False to bypass physical channel entirely

# Channel type settings
# Options: 'awgn', 'rayleigh', 'rician', 'frequency_selective'
CHANNEL_TYPE = 'awgn'

# SNR settings
SNR_DB = 20.0  # Signal-to-noise ratio in dB
SNR_RANGE = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]  # Expanded for adaptive modulation

# Modulation settings
# Options: 'qam', 'psk', 'ofdm'
MODULATION_TYPE = 'qam'
MODULATION_ORDER = 16  # Constellation size (4, 16, 64, 256)
BITS_PER_DIMENSION = 4  # How many bits to use per dimension

# Channel coding settings
USE_CHANNEL_CODING = True
CODING_RATE = 0.75  # Rate 3/4 coding
CODING_TYPE = 'repetition'  # Options: 'repetition', 'ldpc', 'turbo'

# Adaptive modulation settings (NEW)
ENABLE_ADAPTIVE_MODULATION = True
SNR_THRESHOLD_HIGH = 25.0  # Above this, use higher order modulation
SNR_THRESHOLD_LOW = 15.0   # Below this, use lower order modulation

# OFDM parameters (if using OFDM)
OFDM_CARRIERS = 64
OFDM_CP_RATIO = 0.25  # Cyclic prefix ratio

# Fading parameters
RAYLEIGH_VARIANCE = 1.0
RICIAN_K_FACTOR = 4.0  # Higher = more line-of-sight component

# Semantic-specific settings
USE_IMPORTANCE_WEIGHTING = True  # Weight dimensions by importance
WEIGHT_METHOD = 'semantic'  # 'variance', 'pca', 'semantic', or 'uniform'

# Advanced error protection (NEW)
ENABLE_UNEQUAL_ERROR_PROTECTION = True
PROTECTION_LEVELS = 3  # Number of different protection levels

# Experiment settings
SAVE_CHANNEL_PLOTS = True
LOG_CHANNEL_METRICS = True
CHANNEL_RESULTS_DIR = './channel_results'

# Physical layer representation settings
EMBEDDING_SCALING = True  # Scale embeddings to match physical constraints
MAX_DIMENSIONS_PER_BLOCK = 1024  # Group dimensions into blocks for transmission

# DVAE training settings (NEW)
COLLECT_TRANSMISSION_DATA = True
TRANSMISSION_PAIRS_DIR = './transmission_pairs'
MAX_TRANSMISSION_PAIRS = 10000
# Add to your existing configuration
# Advanced compression settings
VAE_COMPRESSION = True
VAE_COMPRESSION_FACTOR = 0.6

# Content-adaptive coding settings
ENABLE_CONTENT_ADAPTIVE_CODING = True
CONTENT_CLASSIFIER_PATH = './models/content_classifier.pth'

# Semantic loss settings
USE_SEMANTIC_LOSS = True