# Configuration file for physical channel settings in semantic communication
# Configured for OFDM with frequency-selective channel

# General physical channel settings
ENABLE_PHYSICAL_CHANNEL = True  # Set to False to bypass physical channel entirely

# Channel type settings
# Options: 'awgn', 'rayleigh', 'rician', 'frequency_selective'
CHANNEL_TYPE = 'frequency_selective'  # Using frequency-selective channel

# Channel characteristics
MULTIPATH_TAPS = 5  # Number of delay taps for frequency-selective channel
TAP_POWER_PROFILE = [0.8, 0.6, 0.4, 0.2, 0.1]  # Power profile for multipath taps (decreasing)
DELAY_SPREAD = 5  # Delay spread in samples

# SNR settings
SNR_DB = 18.0  # Slightly lower SNR for realistic frequency-selective conditions
SNR_RANGE = [5.0, 10.0, 15.0, 20.0, 25.0]  # Range for testing/adaptation

# Modulation settings
# Options: 'qam', 'psk', 'ofdm'
MODULATION_TYPE = 'ofdm'  # Using OFDM modulation
MODULATION_ORDER = 16  # Constellation size (4, 16, 64, 256) for each subcarrier
BITS_PER_DIMENSION = 4  # For 16-QAM

# OFDM-specific parameters
OFDM_CARRIERS = 256  # Increased number of subcarriers for frequency diversity
OFDM_CP_RATIO = 0.25  # Cyclic prefix ratio (25% of symbol length)
OFDM_PILOT_RATIO = 0.1  # 10% of carriers used as pilots for channel estimation
OFDM_NULL_CARRIERS = 8  # Number of null carriers at band edges
OFDM_DC_NULL = True  # Null the DC carrier

# Channel coding settings
USE_CHANNEL_CODING = True
CODING_RATE = 0.5  # More robust coding rate for frequency-selective channel
CODING_TYPE = 'repetition'  # Options: 'repetition', 'ldpc', 'turbo'
USE_INTERLEAVING = True  # Enable interleaving to combat frequency-selective fading

# Adaptive modulation settings
ENABLE_ADAPTIVE_MODULATION = True
SNR_THRESHOLD_HIGH = 22.0  # Above this, use higher order modulation
SNR_THRESHOLD_LOW = 12.0   # Below this, use lower order modulation

# Semantic-specific settings
USE_IMPORTANCE_WEIGHTING = True  # Weight dimensions by importance
WEIGHT_METHOD = 'semantic'  # 'variance', 'pca', 'semantic', or 'uniform'

# Advanced subcarrier allocation
ENABLE_SUBCARRIER_ALLOCATION = True  # Allocate subcarriers based on importance
RESERVED_CARRIERS_RATIO = 0.2  # 20% of carriers reserved for high-importance data

# Advanced error protection
ENABLE_UNEQUAL_ERROR_PROTECTION = True
PROTECTION_LEVELS = 3  # Number of different protection levels

# OFDM equalization method
EQUALIZATION_METHOD = 'mmse'  # Options: 'zf' (zero-forcing), 'mmse' (minimum mean square error)
CHANNEL_ESTIMATION = 'pilot'  # Options: 'pilot', 'preamble', 'decision-directed'

# Content-adaptive coding settings
ENABLE_CONTENT_ADAPTIVE_CODING = True
CONTENT_CLASSIFIER_PATH = './models/content_classifier.pth'

# Channel compensation settings for semantic recovery
FREQUENCY_DOMAIN_EQUALIZATION = True
ENABLE_SOFT_DEMAPPING = True  # Use soft-decision demapping for better performance

# Advanced compression settings
VAE_COMPRESSION = True
VAE_COMPRESSION_FACTOR = 0.5  # Higher compression to fit more data in OFDM symbols

# Experiment settings
SAVE_CHANNEL_PLOTS = True
LOG_CHANNEL_METRICS = True
CHANNEL_RESULTS_DIR = './channel_results'

# Physical layer representation settings
EMBEDDING_SCALING = True
MAX_DIMENSIONS_PER_BLOCK = 512  # Reduced to better fit OFDM structure

# DVAE training settings
COLLECT_TRANSMISSION_DATA = True
TRANSMISSION_PAIRS_DIR = './transmission_pairs'
MAX_TRANSMISSION_PAIRS = 10000

# Semantic loss settings
USE_SEMANTIC_LOSS = True

# Enhanced OFDM-Semantic integration
SEMANTIC_SUBCARRIER_MAPPING = 'importance_based'  # Maps semantic importance to subcarrier quality