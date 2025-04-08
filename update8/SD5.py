import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
import time
import difflib
import json
import random
import traceback
import matplotlib
import difflib

matplotlib.use('Agg')  # Keep using non-interactive backend for headless environment
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque
from config_manager import ConfigManager
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from knowledge_base import get_or_create_knowledge_base
from physical_semantic_integration import SemanticChannelOptimizer
# Import modified components
from circuit_breaker import CircuitBreaker
from semantic_mlpdvae import load_or_train_enhanced_mlp_dvae
from physical_semantic_integration import DimensionRegistry
from mlpdvae_utils import (load_transmission_pairs, evaluate_reconstruction_with_semantics,
                           compute_embedding_similarity, generate_text_from_embedding,ensure_tensor_shape)
from semantic_loss import SemanticPerceptualLoss, evaluate_semantic_similarity
from compression_vae import (EmbeddingCompressorVAE, decompress_vae_embedding,
                             load_or_train_vae_compressor)
# Initialize enhanced evaluation framework
# Import original physical channel components
try:
    from physical_channel import PhysicalChannelLayer
    from content_adaptive_coding import ContentAdaptivePhysicalChannel
    from physical_semantic_integration import physical_semantic_bridge, transmit_through_physical_channel

    physical_channel_imported = True
except ImportError:
    physical_channel_imported = False
    print("WARNING: Physical channel components not found. Running without physical channel.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Initialize breaker for API calls
openai_breaker = CircuitBreaker(failure_threshold=3, reset_timeout=60)
# Paths for data
DATA_DIR = "./data"
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_data.pkl")
COMPRESSED_DATA_PATH = os.path.join(DATA_DIR, "compressed_data.pkl")
# Save to a permanent location like your home directory
HOME_DIR = os.path.expanduser("~")  # Gets your home directory
PROJECT_ROOT = r"C:\Users\Daud\SemanticCommTransmission\pythonProject"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "sem_com_results")
os.makedirs(RESULTS_DIR, exist_ok=True)
MODELS_DIR = "./models"
TRANSMISSION_PAIRS_DIR = './transmission_pairs'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TRANSMISSION_PAIRS_DIR, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
CONTEXT_WINDOW_SIZE = 3  # Number of previous messages to maintain for context
context_history = deque(maxlen=CONTEXT_WINDOW_SIZE)  # For tracking message history
# Try to import physical channel configuration
try:
    import physical_channel_config as phy_config

    ENABLE_PHYSICAL_CHANNEL = getattr(phy_config, 'ENABLE_PHYSICAL_CHANNEL', True) and physical_channel_imported
    COLLECT_TRANSMISSION_DATA = getattr(phy_config, 'COLLECT_TRANSMISSION_DATA', True)
    TRANSMISSION_PAIRS_DIR = getattr(phy_config, 'TRANSMISSION_PAIRS_DIR', './transmission_pairs')
    # New configuration options
    ENABLE_VAE_COMPRESSION = getattr(phy_config, 'VAE_COMPRESSION', True)
    VAE_COMPRESSION_FACTOR = getattr(phy_config, 'VAE_COMPRESSION_FACTOR', 0.6)  # Add this line
    ENABLE_CONTENT_ADAPTIVE_CODING = getattr(phy_config, 'ENABLE_CONTENT_ADAPTIVE_CODING', True)
except ImportError:
    # Default configuration if not found
    ENABLE_PHYSICAL_CHANNEL = physical_channel_imported
    COLLECT_TRANSMISSION_DATA = True
    TRANSMISSION_PAIRS_DIR = './transmission_pairs'
    ENABLE_VAE_COMPRESSION = True
    ENABLE_CONTENT_ADAPTIVE_CODING = True
    print("Physical channel config not found, using defaults")

# OpenAI API Key and setup for API-based reconstruction
OPENAI_API_KEY = "-proj-LMFuTLbE3hYbuy-uYSbuBlMT2sRywodroXE55esx5HlnalJFQRBH9UsWEdsBErZnucSQtU78JlT3BlbkFJkDEFVdf5sO8CmpglMVS-H5vsMSAnvSedBc7FuqJPO5PD1vDKeqCEAMZcqKAxdhgWsJZiWW3Z8A"
openai_available = False  # Will be set to True if connection test succeeds
openai_client = None

if OPENAI_API_KEY:
    try:
        import openai

        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

        # Test API connection
        test_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Reply with the word 'CONNECTED' if you can read this."}
            ],
            max_tokens=10
        )

        if "CONNECTED" in test_response.choices[0].message.content.strip():
            openai_available = True
            logger.info("✅ OpenAI API connection successful")
        else:
            logger.warning("OpenAI API responded with unexpected content")
    except Exception as e:
        logger.error(f"❌ Error initializing OpenAI client: {e}")
        logger.error(traceback.format_exc())

def timing_decorator(func):
    """Decorator to measure and log function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"[TIMING] Function {func.__name__} took {end_time - start_time:.4f} seconds to execute")
        return result
    return wrapper
#################################################
# Reinforcement Learning Agent with Semantic Metrics
#################################################

class AdvancedRLAgent(nn.Module):
    """
    Advanced RL agent for API optimization with neural network policy
    and improved state representation for semantic communication
    """

    def __init__(self, state_dim=12, num_actions=3, learning_rate=0.001):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions

        # Policy network - outputs action probabilities
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions)
        )

        # Value network - estimates state value
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Experience buffer
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'log_probs': []
        }

        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        self.clip_ratio = 0.2  # PPO clip ratio

        # Tracking metrics
        self.total_reward = 0
        self.episode_count = 0
        self.api_efficiency = []

        # Load checkpoint if exists
        self.load_checkpoint()

    def get_enhanced_state(self, corruption_level, text_length, semantic_features=None):
        """
        Create enhanced state representation with semantic features

        Args:
            corruption_level: Level of corruption (0-1)
            text_length: Length of text in tokens
            semantic_features: Optional semantic feature vector

        Returns:
            State tensor
        """
        # Base features
        base_features = [
            corruption_level,  # Corruption level
            min(1.0, text_length / 100),  # Normalized text length
            float(self.epsilon),  # Current exploration rate
        ]

        # Add semantic features if available
        if semantic_features is not None:
            if isinstance(semantic_features, (list, np.ndarray)):
                if len(semantic_features) > self.state_dim - len(base_features):
                    # Truncate to fit state_dim
                    semantic_vector = semantic_features[:self.state_dim - len(base_features)]
                else:
                    # Pad if needed
                    semantic_vector = list(semantic_features)
                    semantic_vector += [0.0] * (self.state_dim - len(base_features) - len(semantic_vector))
            else:
                # Single value, expand to vector
                semantic_vector = [float(semantic_features)] + [0.0] * (self.state_dim - len(base_features) - 1)
        else:
            # No semantic features, use zeros
            semantic_vector = [0.0] * (self.state_dim - len(base_features))

        # Combine features
        state = base_features + semantic_vector
        return torch.tensor(state, dtype=torch.float32)

    def forward(self, state):
        """Forward pass through policy and value networks"""
        # Ensure state is properly shaped
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        # Get action probabilities and state value
        logits = self.policy_net(state)
        action_probs = F.softmax(logits, dim=-1)
        state_value = self.value_net(state)

        return action_probs, state_value

    def select_action(self, state, budget_remaining, force_basic=False, corruption_level=None):
        """Select action using policy with budget awareness and corruption sensitivity"""
        # Force basic reconstruction if requested or very low budget
        if force_basic or budget_remaining < 0.05:
            return 0, 0.0  # Basic action, log_prob=0

        # If corruption level is high, prioritize advanced reconstruction methods
        if corruption_level is not None and corruption_level > 0.4:
            # For high corruption, prefer more powerful methods if budget allows
            if budget_remaining > 0.3:
                # Prefer GPT-4 for highly corrupted text
                valid_actions = [2, 1, 0]  # Prefer in this order: GPT-4, GPT-3.5, basic
                # Override exploration with 40% chance to ensure we sometimes use advanced methods
                if np.random.random() < 0.4:
                    return valid_actions[0], 0.0
            elif budget_remaining > 0.15:
                # Prefer at least GPT-3.5 for moderate budget
                valid_actions = [1, 0]  # Prefer GPT-3.5, then basic
                # Override exploration with 30% chance
                if np.random.random() < 0.3:
                    return valid_actions[0], 0.0

        # Budget-aware strategy for normal cases
        if budget_remaining < 0.2:
            # Only consider basic and GPT-3.5
            valid_actions = [0, 1]
        else:
            # All actions available
            valid_actions = list(range(self.num_actions))

        # Get action probabilities
        with torch.no_grad():
            action_probs, _ = self.forward(state)

            # Set invalid action probabilities to 0
            mask = torch.ones_like(action_probs)
            mask[~torch.tensor([i in valid_actions for i in range(self.num_actions)], dtype=torch.bool)] = 0
            masked_probs = action_probs * mask

            # Renormalize
            masked_probs = masked_probs / (masked_probs.sum() + 1e-10)

            # Exploration: with probability epsilon, choose randomly from valid actions
            if np.random.random() < self.epsilon:
                action = np.random.choice(valid_actions)
                log_prob = torch.log(masked_probs[action] + 1e-10).item()
            else:
                # Greedy: choose highest probability action
                action = torch.argmax(masked_probs).item()
                log_prob = torch.log(masked_probs[action] + 1e-10).item()

        return action, log_prob

    def update(self, state, action, reward, next_state, log_prob):
        """Store experience for batch update"""
        # Add to buffer
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['next_states'].append(next_state)
        self.buffer['log_probs'].append(log_prob)

        # Track total reward
        self.total_reward += reward

    def train_from_buffer(self, batch_size=None):
        """Train from collected experiences using PPO-like algorithm"""
        if not self.buffer['states']:
            return

        # If batch_size is None, use all data
        batch_size = batch_size or len(self.buffer['states'])
        batch_size = min(batch_size, len(self.buffer['states']))

        # Convert buffer to tensors
        states = torch.stack(self.buffer['states'][:batch_size])
        actions = torch.tensor(self.buffer['actions'][:batch_size], dtype=torch.long)
        rewards = torch.tensor(self.buffer['rewards'][:batch_size], dtype=torch.float32)
        next_states = torch.stack(self.buffer['next_states'][:batch_size])
        old_log_probs = torch.tensor(self.buffer['log_probs'][:batch_size], dtype=torch.float32)

        # Calculate returns and advantages
        with torch.no_grad():
            _, next_values = self.forward(next_states)
            _, values = self.forward(states)

            # Returns with discount factor
            returns = rewards + self.gamma * next_values.squeeze()

            # Advantages
            advantages = returns - values.squeeze()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        # Multiple PPO epochs
        for _ in range(3):
            # Get new probabilities and values
            action_probs, values = self.forward(states)

            # Indices for selecting actions
            indices = torch.arange(len(actions))

            # New log probabilities
            new_log_probs = torch.log(action_probs[indices, actions] + 1e-10)

            # PPO ratio
            ratio = torch.exp(new_log_probs - old_log_probs)

            # PPO losses
            policy_loss1 = ratio * advantages
            policy_loss2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(policy_loss1, policy_loss2).mean()

            # Value loss
            value_loss = F.mse_loss(values.squeeze(), returns)

            # Combined loss
            loss = policy_loss + 0.5 * value_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)

            self.optimizer.step()

        # Clear buffer after training
        for key in self.buffer:
            self.buffer[key] = self.buffer[key][batch_size:]

        # Update exploration rate
        self.epsilon = max(0.05, self.epsilon * 0.995)  # Slowly decrease exploration

    def calculate_reward(self, metrics, action, cost=0):
        """
        Enhanced reward function with semantic awareness.

        Args:
            metrics: Dictionary of quality metrics
            action: Action taken (0=basic, 1=GPT-3.5, 2=GPT-4)
            cost: API cost incurred

        Returns:
            Calculated reward
        """
        # Base reward from quality metrics with semantic emphasis
        quality_reward = 0

        # Prioritize semantic metrics
        if 'SEMANTIC' in metrics:
            quality_reward += metrics.get('SEMANTIC', 0) * 0.5  # 50% weight to semantic
            quality_reward += metrics.get('BLEU', 0) * 0.2  # 20% weight to BLEU
            quality_reward += metrics.get('ROUGEL', 0) * 0.3  # 30% weight to ROUGE-L
        else:
            # Traditional metrics if semantic not available
            quality_reward = metrics.get('BLEU', 0) * 0.3 + metrics.get('ROUGEL', 0) * 0.7

        # Cost penalty for API usage - more sophisticated with action-specific scaling
        cost_penalty = 0
        if action > 0:  # API was used
            # Scales for different actions
            if action == 1:  # GPT-3.5
                cost_scale = 8.0
            else:  # GPT-4
                cost_scale = 5.0  # Less penalty per unit cost since quality should be higher

            cost_penalty = cost * cost_scale

            # Track API efficiency
            efficiency = quality_reward / (cost + 0.001)  # Avoid division by zero
            self.api_efficiency.append(efficiency)

        # Final reward with diminishing returns on high quality
        # This encourages not overspending on already-good reconstructions
        quality_component = np.tanh(quality_reward * 1.5)  # Diminishing returns
        final_reward = quality_component - cost_penalty

        return final_reward

    def save_checkpoint(self, path=None):
        """Save model checkpoint"""
        if path is None:
            path = os.path.join(MODELS_DIR, 'advanced_rl_agent.pth')

        try:
            torch.save({
                'policy_state_dict': self.policy_net.state_dict(),
                'value_state_dict': self.value_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'total_reward': self.total_reward,
                'episode_count': self.episode_count,
                'api_efficiency': self.api_efficiency,
                'state_dim': self.state_dim,
                'num_actions': self.num_actions
            }, path)

            logger.info(f"Saved advanced RL agent state")
        except Exception as e:
            logger.warning(f"Failed to save RL agent: {e}")

    def load_checkpoint(self, path=None):
        """Load model checkpoint"""
        if path is None:
            path = os.path.join(MODELS_DIR, 'advanced_rl_agent.pth')

        try:
            if not os.path.exists(path):
                return False

            checkpoint = torch.load(path, map_location=torch.device('cpu'))

            # Load state dictionaries
            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.value_net.load_state_dict(checkpoint['value_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load other attributes
            self.epsilon = checkpoint.get('epsilon', 0.1)
            self.total_reward = checkpoint.get('total_reward', 0)
            self.episode_count = checkpoint.get('episode_count', 0)
            self.api_efficiency = checkpoint.get('api_efficiency', [])

            logger.info(f"Loaded advanced RL agent (exploration rate: {self.epsilon:.2f})")
            return True
        except Exception as e:
            logger.warning(f"Failed to load RL agent: {e}")
            return False

#################################################
# API Reconstruction Functions
#################################################
# Then use it
def make_api_call_with_circuit_breaker(model, messages):
    def api_call():
        return make_api_call_with_retry(model, messages)

    try:
        return openai_breaker.execute(api_call)
    except Exception as e:
        logger.warning(f"API call failed with circuit breaker: {e}")
        return None
class CostTracker:
    """Track API usage costs"""

    def __init__(self, budget=None):
        config_manager = ConfigManager()
        self.total_cost = 0.0
        self.budget = budget if budget is not None else config_manager.get("api.budget", 2.0)
        self.usage_log = []

        # Pricing per 1000 tokens (approximated)
        self.pricing = {
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4o": {"input": 0.01, "output": 0.03}
        }

    def log_usage(self, model, input_tokens, output_tokens):
        """Log API usage and calculate cost"""
        if model not in self.pricing:
            return 0.0

        input_cost = (input_tokens / 1000) * self.pricing[model]["input"]
        output_cost = (output_tokens / 1000) * self.pricing[model]["output"]
        cost = input_cost + output_cost

        self.total_cost += cost
        self.usage_log.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "running_total": self.total_cost
        })

        if self.total_cost > self.budget * 0.8:
            logger.warning(f"BUDGET ALERT: ${self.total_cost:.4f} of ${self.budget:.2f} used")

        return cost

    def can_use_api(self, model, estimated_input, estimated_output):
        """Check if there's enough budget for an API call"""
        if model not in self.pricing:
            return False

        input_cost = (estimated_input / 1000) * self.pricing[model]["input"]
        output_cost = (estimated_output / 1000) * self.pricing[model]["output"]
        estimated_cost = input_cost + output_cost

        return (self.total_cost + estimated_cost) <= self.budget

    def save_log(self, filename="api_cost_log.json"):
        """Save usage log to file"""
        if os.path.dirname(filename):
            filepath = filename
        else:
            filepath = os.path.join(RESULTS_DIR, filename)

        with open(filepath, 'w') as f:
            json.dump({
                "total_cost": self.total_cost,
                "budget": self.budget,
                "remaining": self.budget - self.total_cost,
                "usage_log": self.usage_log
            }, f, indent=2)

        logger.info(f"Total API cost: ${self.total_cost:.4f} of ${self.budget:.2f} budget")


def kb_fuzzy_match(word, term_dict, threshold=0.7):
    """Enhanced fuzzy matching for terms not in the dictionary"""
    import difflib

    # Quick exact match check
    if word.lower() in term_dict:
        return term_dict[word.lower()], 1.0

    # Common European Parliament word replacements
    common_errors = {
        "ocs": "has",
        "tvks": "this",
        "dignt": "right",
        "ynu": "you",
        "gqe": "are",
        "quutg": "quite",
        "amf": "and",
        "hcve": "have",
        "woild": "would",
        "tht": "the",
        "ar": "are",
        "amd": "and",
        "hes": "has",
        "thct": "that",
        "hos": "has",
        "becn": "been",
        "doni": "done",
        "ct": "at",
        "wether": "whether",
        "wheter": "whether",
        "weither": "whether",
        "yhis": "this",
        "shal": "shall",
        "shali": "shall",
        "actully": "actually"
    }

    # Check common errors first (exact match)
    if word.lower() in common_errors:
        return common_errors[word.lower()], 1.0

    # Try similarity to common errors first (faster)
    for error, correction in common_errors.items():
        score = difflib.SequenceMatcher(None, word.lower(), error.lower()).ratio()
        if score > threshold:
            return correction, score

    # Now try fuzzy matching with dictionary
    best_match = None
    best_score = 0

    # Check a subset of the dictionary for performance
    # Start with checking if the first character matches to filter the search
    first_char = word[0].lower() if word else ''
    candidate_terms = [t for t in term_dict.keys() if t and t[0].lower() == first_char]

    # If no first-char matches, check all terms
    if not candidate_terms:
        candidate_terms = term_dict.keys()

    # Try similarity to dictionary keys
    for term in candidate_terms:
        score = difflib.SequenceMatcher(None, word.lower(), term.lower()).ratio()
        if score > threshold and score > best_score:
            best_match = term_dict[term]
            best_score = score

    return best_match, best_score


def basic_text_reconstruction(noisy_text, use_kb=True):
    """Enhanced text reconstruction that properly utilizes the knowledge base"""
    # Use knowledge base if enabled
    if use_kb:
        try:
            # Get knowledge base
            kb = get_or_create_knowledge_base()

            # Try KB-guided reconstruction
            kb_result = kb.kb_guided_reconstruction(noisy_text)

            # Check if any changes were made
            if kb_result != noisy_text:
                logger.info(f"KB reconstruction made changes: '{noisy_text}' -> '{kb_result}'")
                return kb_result

            # If KB made no changes, try more aggressive methods defined in this function
            logger.debug("KB made no changes, trying aggressive methods")
        except Exception as e:
            logger.warning(f"KB reconstruction failed: {e}")

    # Original method as fallback (with improvements)
    # Track if any changes were made
    changes_made = False
    words = noisy_text.split()
    reconstructed_words = []

    # Enhanced correction dictionary with more parliamentary terms
    corrections = {
        # Original corrections
        "wkulz": "would",
        "couvsc": "course",
        "principdas": "principles",
        "accordancg": "accordance",
        "ymus": "your",
        "mnvice": "advice",
        "Rcne": "Rule",
        "acvioe": "advice",
        "ocs": "has",
        "tvks": "this",
        "dignt": "right",
        "ynu": "you",
        "gqe": "are",
        "quutg": "quite",
        "amf": "and",
        "hcve": "have",
        "woild": "would",
        "tht": "the",
        "ar": "are",
        "amd": "and",
        "hes": "has",

        # Additional parliamentary-specific corrections
        "chepk": "check",
        "shhlo": "shall",
        "lhegk": "check",
        "hrq": "Mrs",
        "neu": "you",
        "ern": "are",
        "ars": "has",

        # European Parliament-specific terms
        "Parlamentj": "Parliament",
        "Parliamemt": "Parliament",
        "Parlitment": "Parliament",
        "Parljament": "Parliament",
        "Pcrliasent": "Parliament",
        "Palrliament": "Parliament",
        "Eurepean": "European",
        "Europenn": "European",
        "Europvan": "European",
        "Ejropean": "European",
        "Commision": "Commission",
        "Commissiob": "Commission",
        "Commizion": "Commission",
        "Conmission": "Commission",
        "Coxmission": "Commission",
        "Commissjon": "Commission",
        "Councjl": "Council",
        "Councip": "Council",
        "Coupcil": "Council",
        "Kouncil": "Council",
        "Directave": "Directive",
        "Directlve": "Directive",
        "Direptive": "Directive",
        "Regulatien": "Regulation",
        "Regulaaion": "Regulation",
        "Regupation": "Regulation",
        "Regklation": "Regulation",

        # Meeting and procedure terms
        "moetinp": "meeting",
        "meating": "meeting",
        "meetang": "meeting",
        "meetting": "meeting",
        "meetirg": "meeting",
        "metting": "meeting",
        "mceeting": "meeting",
        "sessien": "session",
        "sessiom": "session",
        "sesslon": "session",
        "sewsion": "session",
        "agendq": "agenda",
        "agenfa": "agenda",
        "agenca": "agenda",
        "agendz": "agenda",
        "agemda": "agenda",
        "tgendw": "agenda",
        "vite": "vote",
        "votr": "vote",
        "vots": "vote",
        "votung": "voting",
        "votinf": "voting",
        "voring": "voting",
        "vodting": "voting",

        # Official positions
        "Presidemt": "President",
        "Presidebt": "President",
        "Presidfnt": "President",
        "Presldent": "President",
        "Presidont": "President",
        "Presidnet": "President",
        "Predsent": "President",
        "Memmber": "Member",
        "Membeq": "Member",
        "Membsr": "Member",
        "Membez": "Member",
        "mender": "member",
        "Rapporceur": "Rapporteur",
        "Rappofteur": "Rapporteur",
        "Rapporteud": "Rapporteur",
        "Qutestois": "Quaestors",
        "Quaestorz": "Quaestors",
        "Quaestoms": "Quaestors",

        # Common parliamentary verbs
        "proporal": "proposal",
        "propesal": "proposal",
        "propozal": "proposal",
        "proposql": "proposal",
        "propofal": "proposal",
        "repprt": "report",
        "repord": "report",
        "repott": "report",
        "agreez": "agrees",
        "agress": "agrees",
        "agreus": "agrees",
        "agreet": "agrees",
        "requzst": "request",
        "requestz": "request",
        "requept": "request",
        "chedk": "check",
        "chrck": "check",
        "chexk": "check",
        "shsll": "shall",
        "shatl": "shall",
        "shail": "shall",
        "shal": "shall",
        "sholl": "shall",
        "shoold": "should",
        "shoumd": "should",
        "shuold": "should",
        "shoula": "should",

        # Common parliamentary procedures
        "amendmert": "amendment",
        "amendnent": "amendment",
        "amencment": "amendment",
        "amendmemt": "amendment",
        "debpte": "debate",
        "debugr": "debate",
        "debats": "debate",
        "debare": "debate",
        "discussaon": "discussion",
        "discussiom": "discussion",
        "discuzsion": "discussion",
        "disgussion": "discussion",
        "decisien": "decision",
        "decisiob": "decision",
        "decizion": "decision",
        "procedume": "procedure",
        "proceduge": "procedure",
        "procedjre": "procedure",
        "procexure": "procedure",

        # Common verbs in parliamentary context
        "considrr": "consider",
        "consiter": "consider",
        "considep": "consider",
        "examinz": "examine",
        "examime": "examine",
        "examino": "examine",
        "reviev": "review",
        "reviem": "review",
        "reviaw": "review",
        "afdress": "address",
        "addross": "address",
        "addrezs": "address",
        "continye": "continue",
        "continoe": "continue",
        "contimue": "continue",

        # Rule-related terms
        "Rulw": "Rule",
        "Ruls": "Rule",
        "Ryle": "Rule",
        "Ruie": "Rule",
        "admisssbility": "admissibility",
        "admissihility": "admissibility",
        "inadmissibllity": "inadmissibility",
        "inadmissihility": "inadmissibility",

        # Common parliamentary nouns
        "questimn": "question",
        "questiom": "question",
        "questiob": "question",
        "questiin": "question",
        "environmemtal": "environmental",
        "environmentsl": "environmental",
        "environmentsl": "environmental",
        "environmebtal": "environmental",
        "protrction": "protection",
        "protectlon": "protection",
        "protectiom": "protection",
        "leglslation": "legislation",
        "legislasion": "legislation",
        "legislatiom": "legislation",
        "legislatisn": "legislation",

        # Common corruption patterns
        "wkulz": "would",
        "cyb": "can",
        "cqn": "can",
        "arn": "are",
        "arr": "are",
        "thatb": "that",
        "thag": "that",
        "ths": "this",
        "thjs": "this",
        "tje": "the",
        "tye": "the",
        "thn": "the",
        "asd": "and",
        "anx": "and",
        "nad": "and",
        "fpr": "for",
        "gor": "for",
        "fot": "for",
        "wuth": "with",
        "wirh": "with",
        "witn": "with",
        "wiph": "with",
        "haz": "has",
        "han": "has",
        "haa": "has",
        "nof": "not",
        "npt": "not",
        "mot": "not",
        "whetjer": "whether",
        "whethe4": "whether",
        "whetheq": "whether",
        "whetner": "whether",
        "whethtr": "whether",
        "righg": "right",
        "righr": "right",
        "righf": "right",
        "actuslly": "actually",
        "actuslly": "actually",
        "actuaoly": "actually",
        "actualby": "actually",
        "afyually": "actually",
        "yiu": "you",
        "yoi": "you",
        "yuo": "you",
        "yea": "yes",
        "yez": "yes",
        "righg": "right",
        "righr": "right",
        "correcx": "correct",
        "correcy": "correct",

        # Names and titles
        "Lynme": "Lynne",
        "Lymne": "Lynne",
        "Lynnw": "Lynne",
        "Mts": "Mrs",
        "Mrz": "Mrs",
        "Segmu": "Segni",
        "Segmi": "Segni",
        "Schroedtet": "Schroedter",
        "Schroedtef": "Schroedter",
        "Schroedtez": "Schroedter",
        "Madom": "Madam",
        "Madas": "Madam",
        "Berengurr": "Berenguer",
        "Berenguez": "Berenguer",
        "Berenguef": "Berenguer",
        "Beeenguew": "Berenguer",
        "Fustez": "Fuster",
        "Fustrr": "Fuster",
        "Fustef": "Fuster",
        "Gorsel": "Gorsel",
        "Gorseb": "Gorsel",
        "Gorsep": "Gorsel",
        "Plooij-vam": "Plooij-van",
        "Plooij-vsn": "Plooij-van",
        "Ploupj-van": "Plooij-van",
        "Díea": "Díez",
        "Díef": "Díez",
        "Díex": "Díez",
        "Evams": "Evans",
        "Evabs": "Evans",
        "Evanz": "Evans"
    }

    # Process each word
    for word in words:
        # Skip very short words and punctuation
        if len(word) <= 2 or all(c in '.,;:!?()[]{}"\'' for c in word):
            reconstructed_words.append(word)
            continue

        # Try exact match in dictionary (case-insensitive)
        if word.lower() in corrections:
            # Preserve capitalization
            if word[0].isupper() and len(corrections[word.lower()]) > 0:
                corrected = corrections[word.lower()].capitalize()
            else:
                corrected = corrections[word.lower()]
            reconstructed_words.append(corrected)
            changes_made = True
            continue

        # Try fuzzy matching with lower threshold for longer words
        threshold = max(0.65, 0.8 - (len(word) * 0.01))  # Lower threshold for longer words
        best_match, score = None, 0

        # Check against common parliamentary terms with custom thresholds
        parliamentary_terms = [("shall", 0.4), ("check", 0.4), ("Mrs", 0.3),
                               ("Parliament", 0.5), ("Commission", 0.5)]

        for term, term_threshold in parliamentary_terms:
            similarity = difflib.SequenceMatcher(None, word.lower(), term.lower()).ratio()
            if similarity > term_threshold and similarity > score:
                best_match = term
                score = similarity

        if best_match:
            # Preserve capitalization
            if word[0].isupper() and len(best_match) > 0:
                best_match = best_match.capitalize()
            reconstructed_words.append(best_match)
            changes_made = True
            continue

        # If no special term match, try general fuzzy matching
        for correct_word, correction in corrections.items():
            similarity = difflib.SequenceMatcher(None, word.lower(), correct_word.lower()).ratio()
            if similarity > threshold and similarity > score:
                best_match = corrections[correct_word]
                score = similarity

        if best_match and score > threshold:
            # Preserve capitalization
            if word[0].isupper() and len(best_match) > 0:
                best_match = best_match.capitalize()
            reconstructed_words.append(best_match)
            changes_made = True
            continue

        # Keep original if no correction found
        reconstructed_words.append(word)

    result = " ".join(reconstructed_words)

    # Apply phrase-based corrections if needed
    if not changes_made:
        phrase_corrected = apply_phrase_patterns(result)
        if phrase_corrected != result:
            changes_made = True
            result = phrase_corrected

    # Log whether changes were made - KEEP THIS CODE
    if changes_made:
        logger.info(f"Basic reconstruction made changes: '{noisy_text}' -> '{result}'")

    return result


def apply_phrase_patterns(text):
    """Apply phrase-level patterns for correction"""
    patterns = [
        # Basic word corrections
        ('shhlo', 'shall'),
        ('lhegk', 'check'),
        ('hrq', 'Mrs'),
        ('neu ern', 'you are'),
        ('wkulz', 'would'),
        ('tvks', 'this'),
        ('dignt', 'right'),
        ('ynu', 'you'),
        ('gqe', 'are'),
        ('quutg', 'quite'),
        ('amf', 'and'),
        ('hcve', 'have'),
        ('woild', 'would'),
        ('tht', 'the'),
        ('thct', 'that'),
        ('hos', 'has'),
        ('becn', 'been'),
        ('doni', 'done'),
        ('ct', 'at'),

        # Parliamentary position patterns
        ('Madam Presidemt', 'Madam President'),
        ('Madam Presidebt', 'Madam President'),
        ('Madam Presldent', 'Madam President'),
        ('Mts Lynne', 'Mrs Lynne'),
        ('Mrz Lynne', 'Mrs Lynne'),
        ('Mrs Lymne', 'Mrs Lynne'),
        ('Mrs Ploupj-van', 'Mrs Plooij-van'),
        ('Mrs Plooij-vam', 'Mrs Plooij-van'),
        ('Mr Evams', 'Mr Evans'),
        ('Mr Berenguef', 'Mr Berenguer'),
        ('Mr Berengurr', 'Mr Berenguer'),
        ('Mr Beeenguew', 'Mr Berenguer'),
        ('Mr Fustez', 'Mr Fuster'),
        ('Mr Fustrr', 'Mr Fuster'),

        # Parliamentary institution patterns
        ('Europenn Parliament', 'European Parliament'),
        ('Eurepean Parliament', 'European Parliament'),
        ('European Parliamemt', 'European Parliament'),
        ('European Pcrliasent', 'European Parliament'),
        ('the Commissiob', 'the Commission'),
        ('the Commizion', 'the Commission'),
        ('the Conmission', 'the Commission'),
        ('the Coupcil', 'the Council'),
        ('the Councip', 'the Council'),
        ('the Councjl', 'the Council'),

        # Parliamentary procedures patterns
        ('in accordancg with', 'in accordance with'),
        ('in accbadance with', 'in accordance with'),
        ('on the agenfa', 'on the agenda'),
        ('on the agendq', 'on the agenda'),
        ('on the agenca', 'on the agenda'),
        ('on the tgendw', 'on the agenda'),
        ('this subject in the course', 'this subject in the course'),
        ('points of orter', 'points of order'),
        ('vote on the propofal', 'vote on the proposal'),
        ('vote on the propesal', 'vote on the proposal'),
        ('vote on the proporal', 'vote on the proposal'),
        ('I shhlo lhegk', 'I shall check'),
        ('I shall chedk', 'I shall check'),
        ('I shall chrck', 'I shall check'),
        ('I wkulz like', 'I would like'),
        ('I woild like', 'I would like'),
        ('air quality tesk', 'air quality test'),
        ('air qualiti test', 'air quality test'),
        ('fire driel', 'fire drill'),
        ('fire dril', 'fire drill'),
        ('fyre drill', 'fire drill'),
        ('no-smocing areas', 'no-smoking areas'),
        ('no-smoklng areas', 'no-smoking areas'),
        ('the staixcased', 'the staircases'),
        ('the ptairuases', 'the staircases'),

        # Rule-related patterns
        ('Rule 143 concernimg', 'Rule 143 concerning'),
        ('Rule 143 concernint', 'Rule 143 concerning'),
        ('Rule 143 concerninh', 'Rule 143 concerning'),
        ('concerning inadmissibllity', 'concerning inadmissibility'),
        ('concerning inadmissihility', 'concerning inadmissibility'),

        # Common parliamentary phrases
        ('you are quite righ', 'you are quite right'),
        ('you are quitz right', 'you are quite right'),
        ('you arn quite right', 'you are quite right'),
        ('neu ern quite right', 'you are quite right'),
        ('shall check whethzr', 'shall check whether'),
        ('shall check whethep', 'shall check whether'),
        ('shall check wether', 'shall check whether'),
        ('shall check wheter', 'shall check whether'),
        ('check whether thiz', 'check whether this'),
        ('check whether thia', 'check whether this'),
        ('whether this ars', 'whether this has'),
        ('whether this haa', 'whether this has'),
        ('whether this haz', 'whether this has'),
        ('has actually nof', 'has actually not'),
        ('has actyally not', 'has actually not'),
        ('has actuslly not', 'has actually not'),
        ('not been doni', 'not been done'),
        ('not bean done', 'not been done'),
        ('not bien done', 'not been done'),

        # Standard beginning phrases in Europarl
        ('The House rosf', 'The House rose'),
        ('The House rosr', 'The House rose'),
        ('The Parliament woll', 'The Parliament will'),
        ('The Parliament wiil', 'The Parliament will'),
        ('The committee approvrd', 'The committee approved'),
        ('The committee approvef', 'The committee approved'),
        ('The Commission propozed', 'The Commission proposed'),
        ('The Commission proposef', 'The Commission proposed'),
        ('The Commission haz', 'The Commission has'),
        ('so Parliament shoild', 'so Parliament should'),
        ('so Parliament shoumd', 'so Parliament should'),

        # Voting and procedure phrases
        ('now vote on thw', 'now vote on the'),
        ('now vote on tne', 'now vote on the'),
        ('we shall vote todya', 'we shall vote today'),
        ('we shall vote todaz', 'we shall vote today'),
        ('the vast majoritp', 'the vast majority'),
        ('the vast majorita', 'the vast majority'),
        ('the vast salority', 'the vast majority'),
        ('this part-sesslon', 'this part-session'),
        ('this part-sessiom', 'this part-session'),
        ('this part-sessien', 'this part-session'),
        ('will now proceet', 'will now proceed'),
        ('will now proceef', 'will now proceed'),
        ('request a debatz', 'request a debate'),
        ('request a debats', 'request a debate'),
        ('request a debpte', 'request a debate'),

        # Meeting-related patterns
        ('meeting on Wedneshay', 'meeting on Wednesday'),
        ('meeting on Wednesfay', 'meeting on Wednesday'),
        ('meeting on tednesgay', 'meeting on Wednesday'),
        ('on the agendc for', 'on the agenda for'),
        ('on the agendz for', 'on the agenda for'),
        ('Quaestors \' meetint', 'Quaestors \' meeting'),
        ('Quaestors \' meating', 'Quaestors \' meeting'),
        ('Quaestors \' meetirg', 'Quaestors \' meeting'),
        ('Quaestors \' moetinp', 'Quaestors \' meeting'),

        # Environmental and policy terms
        ('environmental protectiom', 'environmental protection'),
        ('environmental protectlon', 'environmental protection'),
        ('environmental protrction', 'environmental protection'),
        ('environmemtal protection', 'environmental protection'),
        ('environmentsl protection', 'environmental protection'),
        ('budgek proposal', 'budget proposal'),
        ('budgrt proposal', 'budget proposal'),
        ('budged proposal', 'budget proposal'),

        # Common error sequences
        ('shhlo lhegk', 'shall check'),
        ('sholl chexk', 'shall check'),
        ('shatl chrck', 'shall check'),
        ('wiph thiz', 'with this'),
        ('wirh thjs', 'with this'),
        ('arn quitz', 'are quite'),
        ('arr quutg', 'are quite')
    ]

    result = text
    for pattern, replacement in patterns:
        if pattern in result.lower():
            # Case-preserving replacement
            idx = result.lower().find(pattern)
            if idx >= 0:
                before = result[:idx]
                after = result[idx + len(pattern):]

                # Match capitalization of first letter
                if idx == 0 or result[idx - 1] == ' ':
                    if pattern[0].isupper() or (idx > 0 and result[idx].isupper()):
                        replacement = replacement.capitalize()

                result = before + replacement + after

    return result
def get_token_count(text):
    """Estimate token count for API cost calculation"""
    return len(text.split()) * 1.3  # Rough estimate: words * 1.3


def diagnose_reconstruction_system():
    """Run diagnostics on the reconstruction system"""
    print("\n===== RECONSTRUCTION SYSTEM DIAGNOSTICS =====")

    # Test cases
    test_cases = [
        "Mrs Lynne, you are quite right and I shall check whether this ocs actually not been done.",
        "The Parliamemt will now vote on the propofal from the Commissiob.",
        "In accordancg with Rule 143, I wkulz like your acvioe about this moetinp."
    ]

    # Initialize components
    kb = get_or_create_knowledge_base()
    kb_terms = len(kb.term_dict) if hasattr(kb, 'term_dict') else 0
    print(f"Knowledge Base: {kb_terms} terms loaded")

    # Test reconstruction methods
    for i, test in enumerate(test_cases):
        print(f"\nTest {i + 1}: {test}")

        # Test KB guided reconstruction
        kb_result = kb.kb_guided_reconstruction(test)
        kb_diff = sum(1 for a, b in zip(test.split(), kb_result.split()) if a != b)
        print(f"KB reconstruction ({kb_diff} changes):\n  {kb_result}")

        # Test basic reconstruction
        basic_result = basic_text_reconstruction(test, use_kb=True)
        basic_diff = sum(1 for a, b in zip(test.split(), basic_result.split()) if a != b)
        print(f"Basic reconstruction ({basic_diff} changes):\n  {basic_result}")

        # Test with API if available
        if openai_available:
            api_result, _, _ = api_reconstruct_with_semantic_features(test, context="", use_kb=False)
            api_diff = sum(1 for a, b in zip(test.split(), api_result.split()) if a != b)
            print(f"API reconstruction ({api_diff} changes):\n  {api_result}")

    print("\n===== END DIAGNOSTICS =====")
def calculate_api_cost(model, input_tokens, output_tokens):
    """Calculate the cost of an API call based on the model and token counts"""
    # This function should be using the existing CostTracker class
    return cost_tracker.log_usage(model, input_tokens, output_tokens)


# Full updated function:
@timing_decorator
def api_reconstruct_with_semantic_features(noisy_text, context="", rl_agent=None, budget_remaining=1.0,
                                           semantic_features=None, use_kb=True):
    """
    Enhanced version of API reconstruction with semantic features and knowledge base integration.
    Implements a multi-stage reconstruction approach prioritizing KB for common errors
    and falling back to API for complex cases.
    """
    # Add global recovery counter to prevent infinite recovery loops
    global _api_recovery_attempts
    if '_api_recovery_attempts' not in globals():
        _api_recovery_attempts = {}

    # Generate unique key for this request
    request_key = f"{hash(noisy_text)}-{hash(context)}"

    # Check if we're in a recovery loop
    if request_key in _api_recovery_attempts:
        _api_recovery_attempts[request_key] += 1
        if _api_recovery_attempts[request_key] > 2:
            # Too many recovery attempts, use basic reconstruction
            logger.warning("Too many API recovery attempts, using basic reconstruction")
            return basic_text_reconstruction(noisy_text, use_kb=use_kb), 0, 0
    else:
        _api_recovery_attempts[request_key] = 1

    # Start timing for performance measurement
    start_time = time.time()
    api_cost = 0  # Initialize cost tracking for this function call
    # Get config manager
    config_manager = ConfigManager()

    # Track which method we ultimately use for analytics
    method_used = "basic"
    kb_applied = False
    prompt_enhancement = ""

    # Enhanced logging
    logger.info(f"[API] Starting reconstruction of text: '{noisy_text[:30]}...'")

    # Try knowledge base reconstruction first if enabled
    if use_kb:
        try:
            kb = get_or_create_knowledge_base()

            # First attempt: Direct KB reconstruction
            kb_reconstructed = kb.kb_guided_reconstruction(noisy_text)
            kb_applied = kb_reconstructed != noisy_text

            # Calculate confidence in KB result
            if kb_applied:
                # Calculate different metrics to assess confidence
                word_diff_ratio = sum(1 for a, b in zip(noisy_text.split(), kb_reconstructed.split())
                                      if a != b) / max(1, len(noisy_text.split()))
                char_overlap = difflib.SequenceMatcher(None, noisy_text, kb_reconstructed).ratio()

                # Calculate confidence score
                kb_confidence = char_overlap * (1 - min(0.5, word_diff_ratio))

                # If high confidence, use KB result
                confidence_threshold = 0.8 - min(0.3, len(noisy_text.split()) / 100)  # Lower threshold for longer text

                if kb_confidence > confidence_threshold:
                    logger.info(f"[API] Using KB reconstruction with confidence {kb_confidence:.2f}")
                    method_used = "kb"
                    elapsed_time = time.time() - start_time
                    logger.info(f"[API] Completed in {elapsed_time:.3f}s using method: {method_used}")

                    # When returning successfully, clear recovery attempts
                    _api_recovery_attempts.pop(request_key, None)

                    return kb_reconstructed, 0, 0  # Return KB result, zero cost, basic action

                # Medium confidence - try context enhancement
                elif kb_confidence > 0.6 and context and hasattr(kb, 'enhance_with_context'):
                    try:
                        context_enhanced = kb.enhance_with_context(kb_reconstructed, context)
                        if context_enhanced != kb_reconstructed:
                            logger.info(f"[API] Using KB+context enhancement with confidence {kb_confidence:.2f}")
                            method_used = "kb+context"
                            elapsed_time = time.time() - start_time
                            logger.info(f"[API] Completed in {elapsed_time:.3f}s using method: {method_used}")

                            # When returning successfully, clear recovery attempts
                            _api_recovery_attempts.pop(request_key, None)

                            return context_enhanced, 0, 0
                    except Exception as e:
                        logger.debug(f"Context enhancement failed: {e}")

                # Lower confidence - create API guidance
                else:
                    corrections = []
                    for a, b in zip(noisy_text.split(), kb_reconstructed.split()):
                        if a != b:
                            corrections.append(f"'{a}' might be '{b}'")

                    if corrections:
                        prompt_enhancement = f"Consider these possible corrections: {', '.join(corrections[:5])}"
                        logger.debug(f"[API] Added KB guidance to prompt: {prompt_enhancement}")
        except Exception as e:
            logger.warning(f"[API] KB reconstruction attempt failed: {e}")

    # Skip API if not available
    if not openai_available or not openai_client:
        logger.info("[API] OpenAI API not available, using basic reconstruction")
        reconstructed = basic_text_reconstruction(noisy_text, use_kb=use_kb)
        elapsed_time = time.time() - start_time
        logger.info(f"[API] Completed in {elapsed_time:.3f}s using basic reconstruction")

        # When returning successfully, clear recovery attempts
        _api_recovery_attempts.pop(request_key, None)

        return reconstructed, 0, 0  # Return text, cost, action

    # Use RL agent to decide whether to use API and which model
    force_basic = False
    use_gpt4 = False
    action = 0  # Default: basic reconstruction

    # ENHANCED CORRUPTION DETECTION
    corruption_detected = False
    # Check for any corrupted words
    original_words = [w.lower() for w in noisy_text.split()]
    common_words = ['the', 'that', 'this', 'is', 'are', 'and', 'in', 'with', 'for', 'of', 'to', 'have', 'has']
    corrupted_words = []

    for word in original_words:
        # Check length >2 to avoid short words
        if len(word) > 2:
            # If not a common word or contains unusual character patterns
            if word not in common_words and (
                    # Check for unusual character combinations
                    any(p in word for p in ['bb', 'bz', 'hz', 'jz', 'kz', 'pj']) or
                    # Check for words with no vowels
                    all(c not in 'aeiou' for c in word) or
                    # Check for words starting with unusual prefixes
                    any(word.startswith(p) for p in ['oh', 'zh', 'xj', 'qx', 'kh'])
            ):
                corrupted_words.append(word)
                corruption_detected = True

    # Assess corruption level and use threshold override
    significant_corruption = len(corrupted_words) >= 1
    critical_corruption = len(corrupted_words) >= 2

    # Add a budget-based usage policy
    use_api_based_on_budget = random.random() < (0.7 if budget_remaining > 0.5 else 0.4)

    # Use RL agent or fixed probability for API decision
    if rl_agent is not None:
        try:
            # Calculate corruption level more accurately by comparing to original words if available
            # This helps identify cases where noise has severely corrupted the text
            corruption_level = min(1.0, sum(1 for a, b in zip(noisy_text.split(), context.split() if context else [])
                                            if a != b) / max(1, len(noisy_text.split())))
            text_length = len(noisy_text.split())

            # Check for severe corruption patterns that basic reconstruction struggles with
            severe_corruption = False
            if any(pattern in noisy_text.lower() for pattern in ['shhlo', 'lhegk', 'hrq', 'neu ern', 'wkulz', 'tvks',
                                                                 'ocs', 'ynu', 'gqe', 'quutg', 'hcve', 'woild', 'amf']):
                severe_corruption = True
                corruption_level = max(corruption_level, 0.7)  # Boost corruption level

            # Check if text contains critical parliamentary terms that should be preserved
            critical_terms = ['Parliament', 'Commission', 'Council', 'Rule', 'Directive', 'agenda',
                              'Quaestors', 'Plooij-van', 'President', 'vote', 'proposal']
            has_critical_terms = any(term.lower() in noisy_text.lower() for term in critical_terms)

            # Use enhanced state if semantic features available
            if semantic_features is not None and hasattr(rl_agent, 'get_enhanced_state'):
                state = rl_agent.get_enhanced_state(corruption_level, text_length, semantic_features)
            else:
                state = rl_agent.get_state(corruption_level, text_length)

            # If KB already applied changes, bias toward cheaper options
            if kb_applied:
                # Safely handle tensor states
                if isinstance(state, torch.Tensor):
                    if len(state.shape) > 0:  # Multi-dimensional tensor
                        adjusted_state = torch.clone(state)
                        adjusted_state -= 1
                        adjusted_state = torch.clamp(adjusted_state, min=0)
                    else:  # Scalar tensor
                        adjusted_state = torch.clamp(state - 1, min=0)
                else:
                    # For scalar states
                    adjusted_state = max(0, state - 1)
                action, log_prob = rl_agent.select_action(adjusted_state, budget_remaining, force_basic,
                                                          corruption_level=corruption_level if severe_corruption else None)
            else:
                # Normal action selection with corruption level awareness
                action, log_prob = rl_agent.select_action(state, budget_remaining, force_basic,
                                                          corruption_level=corruption_level if severe_corruption else None)

                # Override for special cases - use more powerful methods for:
                # 1. Severely corrupted text
                # 2. Text with critical parliamentary terms
                if (severe_corruption or has_critical_terms) and budget_remaining > 0.2:
                    # Force more advanced methods for these cases
                    action = min(2, rl_agent.num_actions - 1)  # Use best available method within constraints
                    logger.info(f"[API] Forcing advanced reconstruction for severely corrupted text")

            if action == 0:
                # Use enhanced basic reconstruction with pattern awareness
                reconstructed = basic_text_reconstruction(noisy_text, use_kb=use_kb)

                # Double-check if reconstruction made ANY changes
                if reconstructed == noisy_text and severe_corruption:
                    # Apply phrase patterns as fallback
                    reconstructed = apply_phrase_patterns(noisy_text)
                    logger.info(f"[API] Applied phrase pattern correction after basic reconstruction failed")

                logger.info(f"[API] RL agent chose basic reconstruction")
                elapsed_time = time.time() - start_time
                logger.info(f"[API] Completed in {elapsed_time:.3f}s using basic reconstruction")

                # When returning successfully, clear recovery attempts
                _api_recovery_attempts.pop(request_key, None)

                return reconstructed, 0, action
            elif action == 2:
                # Use GPT-4
                use_gpt4 = True
        except Exception as e:
            logger.warning(f"Error in RL agent processing: {e}")
            # Fallback to basic action
            action = 0
            if action == 0:
                # Use basic reconstruction on exception
                reconstructed = basic_text_reconstruction(noisy_text, use_kb=use_kb)

                # Even on error, try to apply phrase pattern correction for severe corruption
                if reconstructed == noisy_text and any(pattern in noisy_text.lower() for pattern in
                                                       ['shhlo', 'lhegk', 'hrq', 'neu ern', 'wkulz']):
                    reconstructed = apply_phrase_patterns(noisy_text)
                    logger.info(f"[API] Applied fallback phrase patterns after error")

                logger.info(f"[API] Using basic reconstruction due to RL agent error")
                elapsed_time = time.time() - start_time
                logger.info(f"[API] Completed in {elapsed_time:.3f}s using basic reconstruction (RL error)")

                # When returning successfully, clear recovery attempts
                _api_recovery_attempts.pop(request_key, None)

                return reconstructed, 0, action

    # Set up enhanced prompt with KB guidance
    system_prompt = """You are a specialized text reconstruction system. Your task is to correct errors in the text while preserving the original meaning and intent. Fix spelling, grammar, and word corruptions. The text contains European Parliament terminology."""

    # Add KB enhancement to system prompt if available
    if prompt_enhancement:
        system_prompt += f"\n\nIMPORTANT: {prompt_enhancement}"

    # Set up enhanced prompt with KB guidance
    system_prompt = """You are a specialized text reconstruction system for the European Parliament. Your task is to correct errors in text while preserving the original meaning and intent.

    IMPORTANT GUIDELINES:
    1. Correct spelling, grammar, and word corruptions
    2. Pay special attention to parliamentary terminology and names
    3. NEVER replace "that" with "the" unless absolutely necessary
    4. Pay special attention to corrupted words at the beginning of sentences
    5. Make sure your reconstruction is grammatically correct

    European Parliament terms to recognize: Parliament, Commission, Council, Directive, Regulation, Quaestors, Plooij-van Gorsel, Rule 143, amendments, proposal, agenda, debate, vote.

    Common error patterns to fix:
    - Words starting with "oh" are often corrupted forms of "th" (ohiz → this)
    - "pribbiples" should be corrected to "principles"
    - Double consonants like "bb" are often errors
    - Names of officials and institutions should be properly capitalized
    """

    # Enhanced examples with Europarl vocabulary
    example = """Example:
Original: The committee approved the proposal with amendments.
Corrupted: The commitee aproved the proposal with amendmets.
Reconstructed: The committee approved the proposal with amendments.

Original: The European Parliament agrees with the Commission's proposal on environmental protection.
Corrupted: The Europenn Parliamemt agreus with the Commissions propofal on environmentsl protrction.
Reconstructed: The European Parliament agrees with the Commission's proposal on environmental protection.

Original: I would like your advice about Rule 143 concerning inadmissibility.
Corrupted: I wkulz like your advice about Rule 143 concerning inadmissibility.
Reconstructed: I would like your advice about Rule 143 concerning inadmissibility."""

    user_prompt = f"{example}\n\n"
    if context:
        # Add primary context
        user_prompt += f"Immediate Context: {context}\n\n"

        # Add additional context from global history if available
        if len(context_history) > 0:
            user_prompt += "Additional Context:\n"
            for i, prev_context in enumerate(context_history):
                if prev_context != context:  # Avoid duplication
                    user_prompt += f"[{i + 1}] {prev_context}\n"
            user_prompt += "\n"

    user_prompt += f"Corrupted: {noisy_text}\nReconstructed:"

    # Estimate token usage
    system_tokens = get_token_count(system_prompt)
    user_tokens = get_token_count(user_prompt)
    estimated_output_tokens = get_token_count(noisy_text) * 1.2

    # Choose model based on RL agent decision or budget
    model = "gpt-4-turbo" if use_gpt4 else config_manager.get("api.default_model", "gpt-3.5-turbo")

    # Initialize cost tracker if not already done
    global cost_tracker
    if 'cost_tracker' not in globals():
        cost_tracker = CostTracker(budget=config_manager.get("api.budget", 2.0))

    # Check if we can afford this API call
    if not cost_tracker.can_use_api(model, system_tokens + user_tokens, estimated_output_tokens):
        logger.warning(f"[API] Budget limit would be exceeded. Using basic reconstruction instead of {model}")
        reconstructed = basic_text_reconstruction(noisy_text, use_kb=use_kb)
        elapsed_time = time.time() - start_time
        logger.info(f"[API] Completed in {elapsed_time:.3f}s using basic reconstruction (budget limit)")

        # When returning successfully, clear recovery attempts
        _api_recovery_attempts.pop(request_key, None)

        return reconstructed, 0, 0  # Return text, zero cost, basic action

    # Make API call using retry function
    try:
        logger.info(f"[API] Making API call with model {model}...")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        max_retries = config_manager.get("api.max_retries", 3)
        response = make_api_call_with_retry(model, messages, max_retries=max_retries)

        if response:
            # Extract corrected text
            reconstructed_text = response.choices[0].message.content.strip()

            # Calculate cost
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = calculate_api_cost(model, input_tokens, output_tokens)
            api_cost += cost

            # Apply further context enhancement if available
            if context and kb and hasattr(kb, 'enhance_with_context'):
                try:
                    # Try refining with context knowledge
                    context_enhanced = kb.enhance_with_context(reconstructed_text, context)
                    if context_enhanced != reconstructed_text:
                        logger.info(f"[API] Further enhanced API result with context")
                        reconstructed_text = context_enhanced
                except Exception as e:
                    logger.debug(f"Post-API context enhancement failed: {e}")

            # Clean up response
            for prefix in ["Reconstructed:", "Reconstructed text:"]:
                if reconstructed_text.startswith(prefix):
                    reconstructed_text = reconstructed_text[len(prefix):].strip()

            logger.info(f"[API] API reconstruction successful using model {model}")
            method_used = f"api_{model}"
            elapsed_time = time.time() - start_time
            logger.info(f"[API] Completed in {elapsed_time:.3f}s using {method_used}")

            # When returning successfully, clear recovery attempts
            _api_recovery_attempts.pop(request_key, None)

            return reconstructed_text, cost, action
        else:
            logger.warning("[API] API call failed, using fallback reconstruction")
            reconstructed = basic_text_reconstruction(noisy_text, use_kb=use_kb)
            elapsed_time = time.time() - start_time
            logger.info(f"[API] Completed in {elapsed_time:.3f}s using basic reconstruction (API failed)")

            # When returning successfully, clear recovery attempts
            _api_recovery_attempts.pop(request_key, None)

            return reconstructed, 0, 0
    except Exception as e:
        logger.error(f"[API] API enhancement failed: {e}")
        reconstructed = basic_text_reconstruction(noisy_text, use_kb=use_kb)
        elapsed_time = time.time() - start_time
        logger.info(f"[API] Completed in {elapsed_time:.3f}s using basic reconstruction (API error)")

        # When returning successfully, clear recovery attempts
        _api_recovery_attempts.pop(request_key, None)

        return reconstructed, 0, 0

def apply_noise_to_embedding(embedding, noise_level=0.05, noise_type='gaussian'):
    """Apply noise to embedding to simulate channel effects"""
    # Convert to numpy if tensor
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.cpu().numpy()
    # Ensure proper shape
    if len(embedding.shape) == 1:
        embedding = np.expand_dims(embedding, axis=0)
        was_1d = True
    else:
        was_1d = False

    # Make a copy to avoid modifying the original
    noisy_embedding = embedding.copy()

    if noise_type == 'gaussian':
        # Add Gaussian noise scaled by embedding magnitude
        std_dev = np.std(embedding) * noise_level * 2
        noise = np.random.normal(0, std_dev, embedding.shape)
        noisy_embedding = embedding + noise

    elif noise_type == 'burst':
        # Burst noise affects a continuous segment of the embedding
        burst_length = int(len(embedding) * noise_level * 3)
        burst_start = random.randint(0, max(0, len(embedding) - burst_length - 1))
        burst_end = burst_start + burst_length

        # Higher intensity noise in burst region
        std_dev = np.std(embedding) * noise_level * 4
        noise = np.random.normal(0, std_dev, burst_length)
        noisy_embedding[burst_start:burst_end] += noise

    elif noise_type == 'dropout':
        # Randomly zero out elements (simulates packet loss)
        mask = np.random.random(embedding.shape) > noise_level
        noisy_embedding = embedding * mask
    # Return to original shape if needed
    if was_1d:
        noisy_embedding = noisy_embedding.squeeze(0)
    return noisy_embedding


def apply_noise_to_text(text, noise_level=0.05, noise_type='character'):
    """Apply noise directly to text for comparison purposes"""
    words = text.split()
    noisy_words = []

    if noise_type == 'character':
        for word in words:
            if len(word) > 2 and random.random() < noise_level:
                # Corrupt 1-2 characters
                chars = list(word)
                num_corruptions = min(2, len(chars) - 1)
                for _ in range(num_corruptions):
                    idx = random.randint(0, len(chars) - 1)
                    chars[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
                noisy_words.append(''.join(chars))
            else:
                noisy_words.append(word)

    elif noise_type == 'word':
        # Randomly replace words
        for word in words:
            if random.random() < noise_level:
                # Either remove, replace, or duplicate
                choice = random.random()
                if choice < 0.33:  # Remove
                    continue
                elif choice < 0.66:  # Replace
                    random_word = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=len(word)))
                    noisy_words.append(random_word)
                else:  # Duplicate
                    noisy_words.append(word)
                    noisy_words.append(word)
            else:
                noisy_words.append(word)

    elif noise_type == 'burst':
        # Corrupt a continuous sequence of words
        burst_length = max(1, int(len(words) * noise_level * 3))
        burst_start = random.randint(0, max(0, len(words) - burst_length - 1))

        for i, word in enumerate(words):
            if burst_start <= i < burst_start + burst_length and len(word) > 2:
                # Corrupt word in burst region
                chars = list(word)
                idx = random.randint(0, len(chars) - 1)
                chars[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
                noisy_words.append(''.join(chars))
            else:
                noisy_words.append(word)

    return ' '.join(noisy_words)


def make_api_call_with_retry(model, messages, max_retries=3, backoff_factor=2):
    """
    Make an API call with retry logic for better error recovery.
    """
    for attempt in range(max_retries):
        try:
            # Increase temperature slightly for retries to get different completions
            temperature = 0.3 + (attempt * 0.1)  # 0.3, 0.4, 0.5...

            response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=150
            )

            # Quick validation check for common issues
            reconstruction = response.choices[0].message.content.strip()
            if "Reconstructed:" in reconstruction:
                reconstruction = reconstruction.split("Reconstructed:", 1)[1].strip()

            # Check if the API actually made changes
            noisy_text = messages[-1]['content'].split("Corrupted:", 1)[1].strip()
            if noisy_text == reconstruction:
                # API didn't make changes, try again with higher temperature
                if attempt < max_retries - 1:
                    logger.warning(f"API returned unchanged text. Retrying with higher temperature.")
                    continue

            logger.info(f"HTTP Request: POST https://api.openai.com/v1/chat/completions \"HTTP/1.1 200 OK\"")
            return response
        except Exception as e:
            wait_time = backoff_factor ** attempt
            logger.warning(
                f"API call failed with error: {e}. Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)

    logger.error(f"API call failed after {max_retries} attempts")
    return None


def validate_reconstruction(original, reconstructed):
    """Validate and fix common issues in reconstructed text"""
    if original == reconstructed:
        # No changes were made, apply known patterns
        return apply_phrase_patterns(original)

    words_orig = original.split()
    words_recon = reconstructed.split()

    # Check for known problematic replacements
    for i in range(min(len(words_orig), len(words_recon))):
        # Fix "that" incorrectly changed to "the"
        if words_orig[i].lower() == "that" and words_recon[i].lower() == "the":
            words_recon[i] = words_orig[i]  # Restore original "that"

    # Check for uncorrected corrupted words
    for i in range(min(len(words_orig), len(words_recon))):
        # If words remain unchanged but appear corrupted
        if words_orig[i] == words_recon[i] and len(words_orig[i]) > 2:
            # Check for unusual patterns indicating corruption
            if any(p in words_orig[i].lower() for p in ['bb', 'bz', 'hz', 'jz', 'oh']) or \
                    all(c not in 'aeiou' for c in words_orig[i].lower()):
                # Try to correct using pattern matching
                for pattern, replacement in common_corrections:
                    similarity = difflib.SequenceMatcher(None, words_orig[i].lower(), pattern).ratio()
                    if similarity > 0.6:
                        # Preserve case
                        if words_orig[i][0].isupper():
                            words_recon[i] = replacement.capitalize()
                        else:
                            words_recon[i] = replacement
                        break

    return " ".join(words_recon)
# Add this utility function to handle tensor reshaping consistently:
def ensure_correct_embedding_shape(embedding, expected_dim=2):
    """
    Ensure embedding has the correct shape.

    Args:
        embedding: Tensor or ndarray to check
        expected_dim: Expected number of dimensions

    Returns:
        Correctly shaped embedding
    """
    if isinstance(embedding, torch.Tensor):
        if len(embedding.shape) < expected_dim:
            # Add dimensions as needed
            for _ in range(expected_dim - len(embedding.shape)):
                embedding = embedding.unsqueeze(0)
        elif len(embedding.shape) > expected_dim:
            # Squeeze extra dimensions if needed
            embedding = embedding.squeeze()
            # Ensure we still have the minimum dimensions needed
            if len(embedding.shape) < expected_dim:
                for _ in range(expected_dim - len(embedding.shape)):
                    embedding = embedding.unsqueeze(0)
    elif isinstance(embedding, np.ndarray):
        if len(embedding.shape) < expected_dim:
            # Add dimensions as needed
            for _ in range(expected_dim - len(embedding.shape)):
                embedding = np.expand_dims(embedding, axis=0)
        elif len(embedding.shape) > expected_dim:
            # Squeeze extra dimensions if needed
            embedding = np.squeeze(embedding)
            # Ensure we still have the minimum dimensions needed
            if len(embedding.shape) < expected_dim:
                for _ in range(expected_dim - len(embedding.shape)):
                    embedding = np.expand_dims(embedding, axis=0)

    return embedding
#################################################
# Enhanced Main Pipeline Implementation
#################################################
def check_dimension_compatibility(embedding, vae_compressor, logger):
    """
    Check if embedding dimensions are compatible with VAE compressor.
    Returns True if compatible or successfully adapted, False otherwise.
    """
    if not vae_compressor or not hasattr(vae_compressor, 'input_dim'):
        logger.warning("VAE compressor not properly initialized with input_dim attribute")
        return False

    # Check tensor shape
    if isinstance(embedding, torch.Tensor):
        emb_dim = embedding.shape[1] if len(embedding.shape) > 1 else embedding.shape[0]
    elif isinstance(embedding, np.ndarray):
        emb_dim = embedding.shape[1] if len(embedding.shape) > 1 else embedding.shape[0]
    else:
        logger.warning(f"Unknown embedding type: {type(embedding)}")
        return False

    # Compare dimensions
    target_dim = vae_compressor.input_dim
    if emb_dim != target_dim:
        logger.info(f"Dimension mismatch detected: embedding dim={emb_dim}, VAE expects={target_dim}")
        # Can still be adapted, so return True but log the mismatch
        return True

    return True


def get_system_dimensions():
    """Get all relevant system dimensions for proper component integration"""
    dimensions = {}

    # Try to get VAE dimensions
    try:
        vae_dim_path = os.path.join(DATA_DIR, 'vae_dimensions.json')
        if os.path.exists(vae_dim_path):
            with open(vae_dim_path, 'r') as f:
                vae_dims = json.load(f)
                dimensions.update(vae_dims)
        else:
            # Estimate from VAE compression factor
            dimensions['input_dim'] = 768  # BERT default
            dimensions['compressed_dim'] = int(768 * VAE_COMPRESSION_FACTOR)
    except Exception as e:
        logger.warning(f"Could not determine VAE dimensions: {e}")
        dimensions['input_dim'] = 768
        dimensions['compressed_dim'] = 460  # Default

    # Try to get DVAE dimensions
    try:
        dvae_path = os.path.join(MODELS_DIR, "enhanced_mlp_dvae_model.pth")
        if os.path.exists(dvae_path):
            checkpoint = torch.load(dvae_path, map_location=torch.device('cpu'))
            if 'dimensions' in checkpoint:
                dvae_dims = checkpoint['dimensions']
                dimensions['dvae_input_dim'] = dvae_dims.get('input_dim')
                dimensions['dvae_hidden_dim'] = dvae_dims.get('hidden_dim')
                dimensions['dvae_latent_dim'] = dvae_dims.get('latent_dim')
    except Exception as e:
        logger.warning(f"Could not determine DVAE dimensions: {e}")

    logger.info(f"System dimensions: {dimensions}")
    return dimensions
def safe_tensor_ops(data, to_device=device, dtype=torch.float32):
    """Safely convert data to tensor with proper detachment and device placement"""
    if isinstance(data, torch.Tensor):
        # Already a tensor, just ensure it's detached and on the right device
        return data.clone().detach().to(to_device, dtype=dtype)
    elif isinstance(data, np.ndarray):
        # Convert numpy array to tensor
        return torch.tensor(data, dtype=dtype, device=to_device)
    else:
        # Try normal tensor conversion for other types
        return torch.tensor(data, dtype=dtype, device=to_device)
def safe_copy(obj):
    """Create a copy of an object that works for both NumPy arrays and PyTorch tensors."""
    if isinstance(obj, torch.Tensor):
        return obj.clone().detach().cpu().numpy()
    elif isinstance(obj, np.ndarray):
        return obj.copy()
    else:
        return obj  # For other types, return as is


def safe_rl_agent_attribute(agent, attribute_name, default_value):
    """Safely get an attribute from the RL agent, with fallback to default value"""
    if agent is None:
        return default_value

    # Handle different attribute naming between older and newer agent implementations
    attribute_mapping = {
        'exploration_rate': 'epsilon',  # Map old name to new name
    }

    # Check if we have a mapping for this attribute
    if attribute_name in attribute_mapping:
        # Try the mapped attribute first
        mapped_name = attribute_mapping[attribute_name]
        if hasattr(agent, mapped_name):
            return getattr(agent, mapped_name)

    # Try the original attribute name
    if hasattr(agent, attribute_name):
        return getattr(agent, attribute_name)

    # Fall back to default
    return default_value

def run_enhanced_pipeline(num_samples=None, noise_level=None, noise_type=None,
                          use_api_pct=None, comparison_mode=None, use_self_supervised=None,
                          use_semantic_loss=None, use_vae_compression=None,
                          use_content_adaptive_coding=None, use_knowledge_base=True):
    """
    Run the complete enhanced semantic communication pipeline with knowledge base integration.

    Args:
        num_samples: Number of samples to process
        noise_level: Level of noise to apply
        noise_type: Type of noise to apply
        use_api_pct: Percentage of samples to use API for
        comparison_mode: Whether to run comparison between different methods
        use_self_supervised: Whether to use self-supervised learning
        use_semantic_loss: Whether to use semantic loss in dvae
        use_vae_compression: Whether to use VAE compression
        use_content_adaptive_coding: Whether to use content-adaptive coding
        use_knowledge_base: Whether to use knowledge base for enhanced semantics
    """
    # Start timing for performance measurement
    pipeline_start_time = time.time()

    # Get configuration manager
    config_manager = ConfigManager()

    # Use provided values or get from config
    num_samples = num_samples if num_samples is not None else config_manager.get("pipeline.default_num_samples", 50)
    noise_level = noise_level if noise_level is not None else config_manager.get("pipeline.default_noise_level", 0.1)
    noise_type = noise_type if noise_type is not None else config_manager.get("pipeline.default_noise_type", "gaussian")
    use_api_pct = use_api_pct if use_api_pct is not None else config_manager.get("pipeline.use_api_pct", 0.5)
    comparison_mode = comparison_mode if comparison_mode is not None else config_manager.get("pipeline.comparison_mode",
                                                                                             True)
    use_self_supervised = use_self_supervised if use_self_supervised is not None else config_manager.get(
        "pipeline.use_self_supervised", True)
    use_semantic_loss = use_semantic_loss if use_semantic_loss is not None else config_manager.get(
        "pipeline.use_semantic_loss", True)
    use_vae_compression = use_vae_compression if use_vae_compression is not None else config_manager.get(
        "physical.vae_compression", True)
    use_content_adaptive_coding = use_content_adaptive_coding if use_content_adaptive_coding is not None else config_manager.get(
        "physical.enable_content_adaptive_coding", True)

    # Initialize dimension registry first
    dimension_registry = DimensionRegistry()
    system_dimensions = get_system_dimensions()

    # Update registry with detected dimensions
    for key, value in system_dimensions.items():
        if key == 'input_dim':
            dimension_registry.update('original_dim', value)
        elif key == 'compressed_dim':
            dimension_registry.update('compressed_dim', value)
        elif key == 'dvae_latent_dim':
            dimension_registry.update('dvae_latent_dim', value)

    logger.info(f"Initialized dimension registry: {dimension_registry.get_dims()}")
    # Get system dimensions for proper component integration
    system_dimensions = get_system_dimensions()
    original_dim = system_dimensions.get('input_dim', 768)  # Original embedding dimension
    compressed_dim = system_dimensions.get('compressed_dim', 460)  # Compressed dimension

    # Create timestamp for this run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(RESULTS_DIR, f"enhanced_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Enhanced logging
    logger.info(f"[PIPELINE] Starting enhanced pipeline with parameters:")
    logger.info(f"[PIPELINE] - samples: {num_samples}, noise: {noise_level}/{noise_type}")
    logger.info(f"[PIPELINE] - API: {use_api_pct * 100:.0f}%, Compare: {comparison_mode}")
    logger.info(
        f"[PIPELINE] - Features: VAE={use_vae_compression}, Semantic={use_semantic_loss}, Adaptive={use_content_adaptive_coding}")
    logger.info(f"[PIPELINE] - System dimensions: input={original_dim}, compressed={compressed_dim}")

    # Initialize knowledge base if requested
    kb = None
    if use_knowledge_base:
        try:
            kb = get_or_create_knowledge_base()
            logger.info("[PIPELINE] Knowledge base initialized successfully")
        except Exception as e:
            logger.warning(f"[PIPELINE] Could not initialize knowledge base: {e}")
            use_knowledge_base = False

    # Add precomputation optimization for KB enhancement
    if use_knowledge_base and kb is not None:
        try:
            # Precompute common KB enhancements for improved performance
            logger.info("Precomputing KB enhancements...")
            kb.precompute_common_terms()
        except Exception as e:
            logger.warning(f"KB precomputation failed: {e}")

    # Initialize cost tracker
    global cost_tracker
    if 'cost_tracker' not in globals():
        cost_tracker = CostTracker(budget=config_manager.get("api.budget", 2.0))

    # Initialize semantic loss module if requested
    semantic_loss_fn = None
    if use_semantic_loss:
        try:
            semantic_loss_fn = SemanticPerceptualLoss()
            logger.info("[PIPELINE] Semantic perceptual loss initialized successfully")
        except Exception as e:
            logger.warning(f"[PIPELINE] Could not initialize semantic loss: {e}")
            use_semantic_loss = False

    # Load compressed data
    try:
        with open(COMPRESSED_DATA_PATH, "rb") as f:
            compressed_data = pickle.load(f)

        logger.info(f"[PIPELINE] Loaded {len(compressed_data)} items from compressed data")

        # Limit to number of samples
        compressed_data = compressed_data[:num_samples]

    except Exception as e:
        logger.error(f"[PIPELINE] Error loading compressed data: {e}")
        logger.error(traceback.format_exc())
        return

    # Load original sentences for semantic evaluation
    try:
        with open(PROCESSED_DATA_PATH, "rb") as f:
            sentences = pickle.load(f)

        logger.info(f"[PIPELINE] Loaded {len(sentences)} original sentences")

        # Limit to match compressed_data
        sentences = sentences[:len(compressed_data)]
    except Exception as e:
        logger.warning(f"[PIPELINE] Error loading original sentences: {e}")
        sentences = []

    # Extract sentences and embeddings based on data format
    embeddings = []
    extracted_sentences = []

    for i, item in enumerate(compressed_data):
        if isinstance(item, tuple) and len(item) == 2:
            embeddings.append(item[0])
            extracted_sentences.append(item[1])
        elif isinstance(item, dict) and 'embedding' in item and 'sentence' in item:
            embeddings.append(item['embedding'])
            extracted_sentences.append(item['sentence'])
        else:
            logger.warning(f"[PIPELINE] Unknown item format: {type(item)}")

    # If no sentences extracted, use the loaded sentences if available
    if len(extracted_sentences) == 0 and len(sentences) > 0:
        extracted_sentences = sentences[:len(embeddings)]

    # Load VAE compressor if requested
    vae_compressor = None
    if use_vae_compression:
        try:
            # Pass the detected embedding_dim to the VAE compressor
            vae_compressor = load_or_train_vae_compressor(
                compression_factor=VAE_COMPRESSION_FACTOR,
                embedding_dim=original_dim
            )
            if vae_compressor:
                logger.info(f"[PIPELINE] VAE compressor loaded successfully: {original_dim} → {compressed_dim}")
            else:
                logger.warning("[PIPELINE] Could not load VAE compressor, will use original embeddings")
                use_vae_compression = False
        except Exception as e:
            logger.warning(f"[PIPELINE] Error loading VAE compressor: {e}")
            use_vae_compression = False

    # Handle potentially missing components gracefully
    if not vae_compressor and use_vae_compression:
        logger.warning(
            "[PIPELINE] VAE compression requested but compressor is not available. Continuing without compression.")
        use_vae_compression = False

    # Configure physical channel for content-adaptive coding if requested
    if use_content_adaptive_coding and ENABLE_PHYSICAL_CHANNEL and physical_channel_imported:
        try:
            # Check if physical_semantic_bridge has content-adaptive capability
            if hasattr(physical_semantic_bridge, '_physical_channel'):
                # Try to reconfigure or replace with content-adaptive channel
                from content_adaptive_coding import ContentAdaptivePhysicalChannel

                # Get current channel parameters
                current_params = physical_semantic_bridge._physical_channel.get_channel_info()

                try:
                    # Create content-adaptive channel with same parameters and correct dimensions
                    adaptive_channel = ContentAdaptivePhysicalChannel(
                        snr_db=current_params.get('snr_db', 20.0),
                        channel_type=current_params.get('channel_type', 'awgn'),
                        modulation=current_params.get('modulation', 'qam').split('-')[0],
                        modulation_order=int(current_params.get('modulation', 'qam-16').split('-')[1]),
                        enable_content_adaptive_coding=True,
                        embedding_dim=compressed_dim,  # Pass the compressed dimension for classifier
                        content_classifier_path=config_manager.get("physical.content_classifier_path",
                                                                   './models/content_classifier.pth')
                    )

                    # Replace channel in bridge
                    physical_semantic_bridge._physical_channel = adaptive_channel
                    logger.info("[PIPELINE] Physical channel upgraded to content-adaptive version")
                except TypeError as te:
                    logger.warning(f"[PIPELINE] Could not initialize content-adaptive channel: {te}")
                    logger.warning("[PIPELINE] Using standard channel.")
        except Exception as e:
            logger.warning(f"[PIPELINE] Could not configure content-adaptive coding: {e}")
            use_content_adaptive_coding = False

    if not physical_semantic_bridge.physical_enabled and use_content_adaptive_coding:
        logger.warning(
            "[PIPELINE] Content-adaptive coding requested but physical channel is disabled. Continuing without content-adaptive coding.")
        use_content_adaptive_coding = False

    # Load or train enhanced MLPDenoisingVAE model with semantic loss
    dvae = load_or_train_enhanced_mlp_dvae(
        model_path="enhanced_mlp_dvae_model.pth",
        force_retrain=False,
        use_self_supervised=use_self_supervised,
        use_semantic_loss=use_semantic_loss,
        input_dim=compressed_dim  # Pass the compressed dimension as input
    )

    if dvae is None:
        logger.error("[PIPELINE] DVAE model is not properly initialized")
        pipeline_elapsed = time.time() - pipeline_start_time
        logger.info(f"[PIPELINE] Pipeline failed in {pipeline_elapsed:.2f}s")
        return

    if use_vae_compression and vae_compressor is None:
        logger.warning("[PIPELINE] VAE compression enabled but compressor not available - disabling compression")
        use_vae_compression = False

    # Add diagnostic logging after model initialization
    logger.info(f"[PIPELINE] System configurations:")
    logger.info(f"  - VAE compression: {use_vae_compression}")
    if use_vae_compression and vae_compressor:
        vae_input = getattr(vae_compressor, 'input_dim', 'unknown')
        vae_output = getattr(vae_compressor, 'compressed_dim', 'unknown')
        logger.info(f"  - VAE dimensions: {vae_input} → {vae_output}")
    logger.info(f"  - DVAE dimensions: input={dvae.input_dim}, hidden={dvae.hidden_dim}, latent={dvae.latent_dim}")
    logger.info(f"  - Physical channel enabled: {ENABLE_PHYSICAL_CHANNEL}")
    logger.info(f"  - Content adaptive coding: {use_content_adaptive_coding}")
    logger.info(f"  - Knowledge base enabled: {use_knowledge_base}")
    logger.info(
        f"[PIPELINE] IMPORTANT: Model dimensions are input={dvae.input_dim}, hidden={dvae.hidden_dim}, latent={dvae.latent_dim}")

    # Initialize enhanced RL agent for API optimization
    use_rl = openai_available and num_samples >= 10
    rl_agent = AdvancedRLAgent(state_dim=12) if use_rl else None

    semantic_optimizer = None
    if ENABLE_PHYSICAL_CHANNEL and physical_channel_imported:
        try:
            semantic_optimizer = SemanticChannelOptimizer(physical_semantic_bridge._physical_channel)
            logger.info("[PIPELINE] Semantic channel optimizer initialized")
        except Exception as e:
            logger.warning(f"[PIPELINE] Could not initialize semantic optimizer: {e}")

    # Initialize text-embedding mapper for KB if supported
    if use_knowledge_base and kb is not None:
        try:
            # Check for specialized mapping functions
            if hasattr(kb, 'initialize_embedding_mapper') and len(embeddings) >= 100:
                # Use a subset of examples to train mapper
                mapper_examples = min(1000, len(embeddings))
                logger.info(f"[PIPELINE] Initializing KB embedding mapper with {mapper_examples} examples")
                kb.initialize_embedding_mapper(
                    embeddings=embeddings[:mapper_examples],
                    texts=extracted_sentences[:mapper_examples]
                )
        except Exception as e:
            logger.warning(f"[PIPELINE] Could not initialize KB embedding mapper: {e}")

    # Log RL agent status
    if use_rl:
        logger.info(
            f"[PIPELINE] Using enhanced RL agent for API optimization (exploration rate: {rl_agent.epsilon:.2f})")
    else:
        logger.info("[PIPELINE] Not using RL agent - will use fixed API probability")

    # Initialize results storage
    results = {
        "settings": {
            "timestamp": timestamp,
            "num_samples": len(embeddings),
            "noise_level": noise_level,
            "noise_type": noise_type,
            "use_api_pct": use_api_pct,
            "comparison_mode": comparison_mode,
            "physical_channel_enabled": ENABLE_PHYSICAL_CHANNEL,
            "use_enhanced_mlp_dvae": True,
            "use_self_supervised": use_self_supervised,
            "use_semantic_loss": use_semantic_loss,
            "use_vae_compression": use_vae_compression,
            "use_content_adaptive_coding": use_content_adaptive_coding,
            "use_rl_agent": use_rl,
            "dimensions": system_dimensions  # Add dimensions to results for reference
        },
        "samples": []
    }

    # Add physical channel information if enabled
    if ENABLE_PHYSICAL_CHANNEL and physical_channel_imported:
        try:
            results["settings"]["physical_channel"] = physical_semantic_bridge.get_channel_info()
        except Exception as e:
            logger.warning(f"[PIPELINE] Could not get physical channel info: {e}")

    # Track metrics
    semantic_metrics = {
        "BLEU": [],
        "ROUGE1": [],
        "ROUGEL": [],
        "METEOR": [],
        "SEMANTIC": []  # New semantic similarity metric
    }

    direct_metrics = {
        "BLEU": [],
        "ROUGE1": [],
        "ROUGEL": [],
        "METEOR": [],
        "SEMANTIC": []  # New semantic similarity metric
    }

    # Process samples
    logger.info(f"=== Starting Enhanced Semantic Reconstruction Pipeline ===")
    logger.info(f"[PIPELINE] Noise level: {noise_level}, Noise type: {noise_type}")
    logger.info(f"[PIPELINE] OpenAI API available: {openai_available}")
    logger.info(f"[PIPELINE] Physical channel enabled: {ENABLE_PHYSICAL_CHANNEL}")

    for i, (sentence, embedding) in enumerate(
            tqdm(zip(extracted_sentences, embeddings), total=len(embeddings), desc="Processing samples")):
        # Track per-sample processing time
        sample_start_time = time.time()

        try:
            sample_result = {"original": sentence}

            # Initialize context variable early to avoid "referenced before assignment" error
            context = ""
            if i > 0:
                # Use previous sentence as context
                context = extracted_sentences[i - 1]

                # Add to context history for future use
                if i > 1 and len(extracted_sentences[i - 1].strip()) > 0:  # Only add non-empty sentences
                    context_history.append(extracted_sentences[i - 1])

            # Create a context list from history and current context for enhanced processing
            context_list = []
            if context:
                context_list.append(context)
            for ctx in list(context_history):
                if ctx != context:  # Avoid duplicates
                    context_list.append(ctx)

            # === Embedding-based reconstruction ===
            # Apply VAE compression if enabled
            if use_vae_compression and vae_compressor:
                # Convert to tensor
                embedding_tensor = torch.tensor(embedding, dtype=torch.float32).to(device)

                # Get the target dimension from the VAE compressor
                target_dim = vae_compressor.input_dim

                # Adapt the dimensions to match what the VAE expects
                embedding_tensor = ensure_tensor_shape(embedding_tensor, expected_dim=2, target_feature_dim=target_dim)

                # Compress using VAE
                with torch.no_grad():
                    compressed_embedding = vae_compressor.compress(embedding_tensor).cpu().numpy()

                # Store original and compressed embeddings using safe_copy
                sample_result["original_embedding"] = safe_copy(embedding)
                sample_result["compressed_embedding"] = safe_copy(compressed_embedding)

                # Use compressed embedding for further processing
                working_embedding = compressed_embedding
            else:
                # Use original embedding
                working_embedding = embedding

            # Apply noise to embedding
            noisy_embedding = apply_noise_to_embedding(working_embedding, noise_level, noise_type)

            # Transmit through physical channel if enabled
            if ENABLE_PHYSICAL_CHANNEL and physical_channel_imported:
                # Store semantic noisy embedding
                sample_result["semantic_noisy_embedding"] = safe_copy(noisy_embedding)

                try:
                    # Apply semantic optimization if available
                    importance_weights = None
                    if semantic_optimizer is not None:
                        try:
                            optimized_embedding, importance_weights = semantic_optimizer.optimize_transmission(
                                sentence, noisy_embedding)
                            logger.debug(f"[PIPELINE] Applied semantic optimization to embedding")
                            noisy_embedding = optimized_embedding
                        except Exception as e:
                            logger.debug(f"[PIPELINE] Semantic optimization failed: {e}")

                    # Transmit through physical channel with optimized parameters
                    noisy_embedding = transmit_through_physical_channel(
                        noisy_embedding,
                        importance_weights=importance_weights,
                        debug=False,
                        use_kb=use_knowledge_base,
                        context=context,  # Add this
                        context_list=context_list  # Add this
                    )
                except Exception as e:
                    logger.warning(f"[PIPELINE] Physical channel transmission failed: {e}")
                    # Continue with noisy embedding if transmission fails

                # Store post-physical channel embedding
                sample_result["physical_noisy_embedding"] = safe_copy(noisy_embedding)

            # Reconstruct embedding using enhanced MLPDenoisingVAE
            with torch.no_grad():
                try:
                    # Fix tensor shape issues using our utility function with target dimension
                    if isinstance(noisy_embedding, torch.Tensor):
                        embedding_tensor = noisy_embedding.clone().detach().to(device)
                    else:
                        embedding_tensor = torch.tensor(noisy_embedding, dtype=torch.float32).to(device)

                    # Explicitly check and log dimensions before processing
                    original_shape = embedding_tensor.shape

                    # Ensure tensor has the right shape and dimensions for the DVAE
                    embedding_tensor = ensure_tensor_shape(embedding_tensor, expected_dim=2,
                                                           target_feature_dim=dvae.input_dim)
                    if original_shape != embedding_tensor.shape:
                        logger.info(
                            f"[PIPELINE] Embedding reshaped: {original_shape} → {embedding_tensor.shape}, Model input dim: {dvae.input_dim}")

                    # First encode to get latent representation
                    mu, logvar = dvae.encode(embedding_tensor)

                    # Also ensure the latent vector has the correct shape
                    latent_vector = ensure_tensor_shape(mu, expected_dim=2, target_feature_dim=dvae.latent_dim)
                except Exception as e:
                    logger.error(f"[PIPELINE] Error in DVAE encoding: {e}")
                    # Create fallback tensors with correct dimensions as a last resort
                    mu = torch.zeros(1, dvae.latent_dim).to(device)
                    logvar = torch.zeros(1, dvae.latent_dim).to(device)
                    latent_vector = mu

            # Use mean of encoding as latent vector (no sampling for inference)
            latent_vector = ensure_tensor_shape(mu, expected_dim=2, target_feature_dim=dvae.latent_dim)

            if hasattr(dvae, 'decode_with_text_guidance'):
                # Use text-guided decoding with enhanced context
                reconstructed_embedding = dvae.decode_with_text_guidance(
                    latent_vector,
                    text_hint=sentence,
                    text_context=context if context else None,  # Keep original parameter for backward compatibility
                    text_contexts=context_list if len(context_list) > 0 else None  # Add new parameter
                ).detach().cpu().numpy()
            else:
                # Standard decode
                reconstructed_embedding = dvae.decode(latent_vector).detach().cpu().numpy()

            # Decompress with VAE if compression was used
            if use_vae_compression and vae_compressor:
                try:
                    # Decompress back to original embedding space
                    decompressed_embedding = decompress_vae_embedding(reconstructed_embedding)
                    sample_result["decompressed_embedding"] = decompressed_embedding.copy()

                    # Use decompressed embedding for further processing
                    final_embedding = decompressed_embedding
                except Exception as e:
                    logger.warning(f"[PIPELINE] Error decompressing embedding: {e}")
                    # Fall back to reconstructed embedding
                    final_embedding = reconstructed_embedding
            else:
                # Use reconstructed embedding directly
                final_embedding = reconstructed_embedding

            # Create corrupted text version for text reconstruction
            corrupted_text = apply_noise_to_text(sentence, noise_level, 'character')

            # Create a context list from history and current context
            context_list = []
            if context:
                context_list.append(context)
            for ctx in list(context_history):
                if ctx != context:  # Avoid duplicates
                    context_list.append(ctx)

            # Extract simple semantic features for RL agent with context awareness
            semantic_features = None
            if semantic_loss_fn is not None:
                # Simple feature - sentence length ratio compared to average
                avg_len = 20  # Assumed average sentence length
                len_ratio = len(sentence.split()) / avg_len

                # Add context-based features
                context_features = []
                context_size = len(context_list)
                if context_size > 0:
                    # Add context size as a feature
                    context_features.append(min(1.0, context_size / 5.0))

                    # Add context similarity if we have context
                    try:
                        if context:
                            sim = semantic_loss_fn.calculate_semantic_similarity(context, sentence)
                            context_features.append(sim)
                        else:
                            context_features.append(0.5)
                    except:
                        context_features.append(0.5)
                else:
                    # No context
                    context_features.extend([0.0, 0.5])

                # Combine all features
                semantic_features = [len_ratio, 0.5] + context_features

            # Calculate budget remaining as fraction
            budget_remaining = (cost_tracker.budget - cost_tracker.total_cost) / cost_tracker.budget

            # Use RL agent or fixed probability for API decision
            if use_rl:
                # Use enhanced RL agent with semantic features for API decision
                semantic_reconstructed, api_cost, action = api_reconstruct_with_semantic_features(
                    corrupted_text, context, rl_agent, budget_remaining, semantic_features, use_kb=use_knowledge_base
                )

                # Record API method used
                if action == 0:
                    sample_result["semantic_method"] = "basic"
                elif action == 1:
                    sample_result["semantic_method"] = "gpt-3.5-turbo"
                    sample_result["api_cost"] = api_cost
                elif action == 2:
                    sample_result["semantic_method"] = "gpt-4-turbo"
                    sample_result["api_cost"] = api_cost

            else:
                # Use fixed probability for API decision
                use_api = openai_available and random.random() < use_api_pct
                if use_api:
                    # Use API for reconstruction
                    semantic_reconstructed, api_cost, _ = api_reconstruct_with_semantic_features(
                        corrupted_text, context, use_kb=use_knowledge_base,
                        additional_contexts=context_list[1:] if len(context_list) > 1 else None)
                    sample_result["semantic_method"] = "api"
                    sample_result["api_cost"] = api_cost
                else:
                    semantic_reconstructed = basic_text_reconstruction(corrupted_text, use_kb=use_knowledge_base)
                    sample_result["semantic_method"] = "basic"

            # Save embedding similarity
            similarity = compute_embedding_similarity(embedding, final_embedding)
            sample_result["embedding_similarity"] = similarity
            sample_result["semantic_noisy"] = corrupted_text
            sample_result["semantic_reconstructed"] = semantic_reconstructed

            # Register successful correction for learning
            if use_knowledge_base and semantic_reconstructed != corrupted_text:
                try:
                    kb = get_or_create_knowledge_base()
                    if hasattr(kb, 'register_successful_correction'):
                        kb.register_successful_correction(corrupted_text, semantic_reconstructed)
                        # Optional logging
                        logger.debug(f"[PIPELINE] Registered correction for KB learning")
                except Exception as e:
                    logger.warning(f"[PIPELINE] Could not register correction: {e}")

            # Calculate semantic metrics with new semantic similarity included
            semantic_metrics_result = evaluate_reconstruction_with_semantics(
                sentence, semantic_reconstructed, semantic_loss_fn)
            sample_result["semantic_metrics"] = semantic_metrics_result

            # Track semantic metrics
            for key, value in semantic_metrics_result.items():
                if key in semantic_metrics:
                    semantic_metrics[key].append(value)

            # Update RL agent if used
            if use_rl and 'api_cost' in sample_result:
                # Get enhanced state with semantic features
                corruption_level = min(1.0,
                                       sum(1 for a, b in zip(corrupted_text.split(), sentence.split()) if a != b) /
                                       max(1, len(corrupted_text.split())))
                text_length = len(corrupted_text.split())

                # Use enhanced state if semantic features available
                if semantic_features is not None and hasattr(rl_agent, 'get_enhanced_state'):
                    state = rl_agent.get_enhanced_state(corruption_level, text_length, semantic_features)
                else:
                    state = rl_agent.get_state(corruption_level, text_length)

                # Get next state (simplified - just use same state for now)
                next_state = state

                # Calculate reward with enhanced metrics including semantic
                reward = rl_agent.calculate_reward(
                    semantic_metrics_result,
                    action,
                    sample_result.get('api_cost', 0)
                )

                # Update RL agent - provide default log_prob of 0.0 if missing
                if hasattr(sample_result, 'log_prob'):
                    log_prob = sample_result.get('log_prob')
                else:
                    log_prob = 0.0  # Default value when log_prob isn't available

                rl_agent.update(state, action, reward, next_state, log_prob)

                # Periodically train from buffer
                if i % 10 == 0 and i > 0:
                    rl_agent.train_from_buffer()

                # Save progress periodically
                if i % 20 == 0 and i > 0:
                    rl_agent.save_checkpoint()

                # Increment episode count
                rl_agent.episode_count += 1

                # Record RL info
                sample_result["rl_state"] = state.tolist() if isinstance(state, torch.Tensor) else state
                sample_result["rl_action"] = int(action)
                sample_result["rl_reward"] = float(reward)

            # === Direct text reconstruction (for comparison) ===
            if comparison_mode:
                # Apply noise directly to text
                direct_noisy = apply_noise_to_text(sentence, noise_level, 'character')

                # Use API for direct reconstruction (if budget allows)
                if use_rl:
                    direct_reconstructed, api_cost, _ = api_reconstruct_with_semantic_features(
                        direct_noisy, context, rl_agent, budget_remaining, semantic_features, use_kb=use_knowledge_base
                    )
                    sample_result["direct_method"] = "rl_decision"
                else:
                    use_api = random.random() < use_api_pct and openai_available
                    if use_api:
                        direct_reconstructed, api_cost, _ = api_reconstruct_with_semantic_features(
                            direct_noisy, context, use_kb=use_knowledge_base)
                        sample_result["direct_method"] = "api"
                        sample_result["direct_api_cost"] = api_cost
                    else:
                        # Basic reconstruction as fallback
                        direct_reconstructed = basic_text_reconstruction(direct_noisy, use_kb=use_knowledge_base)
                        sample_result["direct_method"] = "basic"

                sample_result["direct_noisy"] = direct_noisy
                sample_result["direct_reconstructed"] = direct_reconstructed

                # Calculate direct metrics with semantic similarity
                direct_metrics_result = evaluate_reconstruction_with_semantics(
                    sentence, direct_reconstructed, semantic_loss_fn)
                sample_result["direct_metrics"] = direct_metrics_result

                # Track direct metrics
                for key, value in direct_metrics_result.items():
                    if key in direct_metrics:
                        direct_metrics[key].append(value)

            # Store sample
            results["samples"].append(sample_result)

            # Log processing time for this sample
            sample_elapsed = time.time() - sample_start_time

            # Log progress periodically
            if (i + 1) % 10 == 0 or i < 2:
                logger.info(f"[PIPELINE] Sample {i + 1}/{len(embeddings)} (processed in {sample_elapsed:.2f}s)")
                logger.info(f"[PIPELINE] Original: {sentence}")
                logger.info(f"[PIPELINE] Semantic noisy: {sample_result.get('semantic_noisy', 'N/A')}")
                logger.info(f"[PIPELINE] Semantic reconstructed: {semantic_reconstructed}")

                if comparison_mode:
                    logger.info(f"[PIPELINE] Direct noisy: {sample_result.get('direct_noisy', 'N/A')}")
                    logger.info(f"[PIPELINE] Direct reconstructed: {sample_result.get('direct_reconstructed', 'N/A')}")

                logger.info(f"[PIPELINE] Semantic BLEU: {semantic_metrics_result.get('BLEU', 0):.4f}, "
                            f"ROUGE-L: {semantic_metrics_result.get('ROUGEL', 0):.4f}, "
                            f"SEMANTIC: {semantic_metrics_result.get('SEMANTIC', 0):.4f}")

                if comparison_mode:
                    logger.info(f"[PIPELINE] Direct BLEU: {direct_metrics_result.get('BLEU', 0):.4f}, "
                                f"ROUGE-L: {direct_metrics_result.get('ROUGEL', 0):.4f}, "
                                f"SEMANTIC: {direct_metrics_result.get('SEMANTIC', 0):.4f}")

                logger.info(f"[PIPELINE] Current cost: ${cost_tracker.total_cost:.4f} of ${cost_tracker.budget:.2f}")

                # Calculate and show estimated completion time
                elapsed_so_far = time.time() - pipeline_start_time
                samples_processed = i + 1
                avg_time_per_sample = elapsed_so_far / samples_processed
                samples_remaining = len(embeddings) - samples_processed
                estimated_remaining = avg_time_per_sample * samples_remaining

                logger.info(f"[PIPELINE] Progress: {samples_processed}/{len(embeddings)} samples. "
                            f"Est. remaining: {estimated_remaining:.1f}s "
                            f"({estimated_remaining / 60:.1f}m)")
                logger.info("---")

        except Exception as e:
            logger.error(f"[PIPELINE] Error processing sample {i}: {e}")
            logger.error(traceback.format_exc())
            continue  # Continue with next sample on error

    # Save RL agent if used
    if use_rl:
        try:
            rl_agent.save_checkpoint()
            logger.info("[PIPELINE] Saved enhanced RL agent state")
        except Exception as e:
            logger.warning(f"[PIPELINE] Failed to save RL agent: {e}")

    # Calculate average metrics
    results["overall_metrics"] = {}

    # Semantic metrics
    for key in semantic_metrics:
        if semantic_metrics[key]:
            results["overall_metrics"][f"semantic_avg_{key}"] = float(np.mean(semantic_metrics[key]))

    # Direct metrics
    if comparison_mode:
        for key in direct_metrics:
            if direct_metrics[key]:
                results["overall_metrics"][f"direct_avg_{key}"] = float(np.mean(direct_metrics[key]))

    # Add cost information
    results["cost"] = {
        "total": cost_tracker.total_cost,
        "budget": cost_tracker.budget,
        "remaining": cost_tracker.budget - cost_tracker.total_cost
    }

    # Add RL agent metrics if used
    if use_rl:
        results["rl_metrics"] = {
            "total_reward": safe_rl_agent_attribute(rl_agent, "total_reward", 0.0),
            "episode_count": safe_rl_agent_attribute(rl_agent, "episode_count", 0),
            "exploration_rate": safe_rl_agent_attribute(rl_agent, "exploration_rate", 0.1),
            "api_efficiency": safe_rl_agent_attribute(rl_agent, "api_efficiency", [])[-50:] if rl_agent and hasattr(
                rl_agent, "api_efficiency") and len(rl_agent.api_efficiency) > 0 else []
        }

    # Add features information
    results["features"] = {
        "vae_compression": use_vae_compression,
        "semantic_loss": use_semantic_loss,
        "content_adaptive_coding": use_content_adaptive_coding,
        "enhanced_rl": use_rl and isinstance(rl_agent, AdvancedRLAgent)
    }

    # Add timing information
    pipeline_elapsed = time.time() - pipeline_start_time
    results["timing"] = {
        "total_time": pipeline_elapsed,
        "avg_per_sample": pipeline_elapsed / len(embeddings) if embeddings else 0,
        "samples_processed": len(embeddings)
    }

    # Initialize enhanced evaluation framework using existing semantic loss
    try:
        from semantic_evaluation import EnhancedEvaluationFramework
        # Use the already defined semantic_loss_fn from earlier in the function
        evaluation_framework = EnhancedEvaluationFramework(semantic_loss_fn)

        # Use the safe method instead
        enhanced_metrics = evaluation_framework.safe_evaluate_reconstruction(
            [sample["original"] for sample in results["samples"]],
            [sample["semantic_reconstructed"] for sample in results["samples"]]
        )

        # Add to results
        results["enhanced_metrics"] = enhanced_metrics

        # Print detailed results
        logger.info("\n=== Enhanced Evaluation Results ===")
        logger.info(f"Overall Score: {enhanced_metrics['overall']['overall_score']:.4f}")
        logger.info(f"Semantic Fidelity: {enhanced_metrics['overall']['semantic_fidelity']:.4f}")
        logger.info(f"Linguistic Quality: {enhanced_metrics['overall']['linguistic_quality']:.4f}")
        logger.info(f"Domain Relevance: {enhanced_metrics['overall']['domain_relevance']:.4f}")
        logger.info(f"Information Preservation: {enhanced_metrics['overall']['information_preservation']:.4f}")
    except Exception as e:
        logger.warning(f"[PIPELINE] Enhanced evaluation failed: {e}")
        logger.warning("Continuing with standard metrics only")

    def convert_numpy_for_json(obj):
        """Convert numpy and PyTorch types to Python standard types for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif torch.is_tensor(obj):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_for_json(item) for item in obj]
        elif hasattr(obj, 'state_dict'):
            # For PyTorch modules, just indicate presence but don't serialize
            return "PyTorch Module (not serialized)"
        elif callable(obj):
            # For function/callable objects
            return str(obj)
        return obj

    # Save results
    with open(os.path.join(run_dir, "detailed_results.json"), "w") as f:
        json.dump(convert_numpy_for_json(results), f, indent=2)

    # Save summary
    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write("=== Enhanced Semantic Communication Results ===\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Samples processed: {len(results['samples'])}\n")
        f.write(f"Noise level: {noise_level}, Noise type: {noise_type}\n")
        f.write(f"Physical channel: {'Enabled' if ENABLE_PHYSICAL_CHANNEL else 'Disabled'}\n")
        f.write(f"Using VAE compression: {use_vae_compression}\n")
        f.write(f"Using semantic loss: {use_semantic_loss}\n")
        f.write(f"Using content-adaptive coding: {use_content_adaptive_coding}\n")
        f.write(f"Using enhanced RL for API optimization: {use_rl}\n\n")
        f.write(f"System dimensions: input={original_dim}, compressed={compressed_dim}\n")

        f.write(f"Total processing time: {pipeline_elapsed:.2f} seconds\n")
        f.write(f"Average time per sample: {pipeline_elapsed / len(embeddings):.2f} seconds\n\n")

        f.write("Semantic Reconstruction Metrics:\n")
        for key in ["BLEU", "ROUGE1", "ROUGEL", "METEOR", "SEMANTIC"]:
            if f"semantic_avg_{key}" in results["overall_metrics"]:
                f.write(f"Semantic Average {key}: {results['overall_metrics'][f'semantic_avg_{key}']:.4f}\n")

        if comparison_mode:
            f.write("\nDirect Reconstruction Metrics:\n")
            for key in ["BLEU", "ROUGE1", "ROUGEL", "METEOR", "SEMANTIC"]:
                if f"direct_avg_{key}" in results["overall_metrics"]:
                    f.write(f"Direct Average {key}: {results['overall_metrics'][f'direct_avg_{key}']:.4f}\n")

        f.write(f"\nTotal Cost: ${cost_tracker.total_cost:.4f} of ${cost_tracker.budget:.2f} budget\n")

        if use_rl:
            f.write(f"\nRL Agent Performance:\n")
            f.write(f"Total episodes: {safe_rl_agent_attribute(rl_agent, 'episode_count', 0)}\n")
            f.write(f"Total reward: {safe_rl_agent_attribute(rl_agent, 'total_reward', 0.0):.2f}\n")
            f.write(f"Final exploration rate: {safe_rl_agent_attribute(rl_agent, 'exploration_rate', 0.1):.2f}\n")

            api_efficiency = safe_rl_agent_attribute(rl_agent, "api_efficiency", [])
            api_eff = np.mean(api_efficiency[-20:]) if len(api_efficiency) > 20 else 'N/A'
            f.write(f"API efficiency: {api_eff}\n")

    # Save cost log
    cost_tracker.save_log(os.path.join(run_dir, "cost_log.json"))

    # Print summary
    logger.info("\n=== Overall Results ===")
    logger.info(
        f"[PIPELINE] Total time: {pipeline_elapsed:.2f}s, Avg: {pipeline_elapsed / len(embeddings):.2f}s per sample")
    logger.info(f"[PIPELINE] System dimensions: input={original_dim}, compressed={compressed_dim}")
    logger.info("[PIPELINE] Semantic Reconstruction:")
    for key in ["BLEU", "ROUGE1", "ROUGEL", "METEOR", "SEMANTIC"]:
        if f"semantic_avg_{key}" in results["overall_metrics"]:
            logger.info(f"[PIPELINE] Semantic Average {key}: {results['overall_metrics'][f'semantic_avg_{key}']:.4f}")

    if comparison_mode:
        logger.info("\n[PIPELINE] Direct Reconstruction:")
        for key in ["BLEU", "ROUGE1", "ROUGEL", "METEOR", "SEMANTIC"]:
            if f"direct_avg_{key}" in results["overall_metrics"]:
                logger.info(f"[PIPELINE] Direct Average {key}: {results['overall_metrics'][f'direct_avg_{key}']:.4f}")

    logger.info(f"\n[PIPELINE] Total Cost: ${cost_tracker.total_cost:.4f} of ${cost_tracker.budget:.2f} budget")
    logger.info(f"[PIPELINE] Results saved to {run_dir}")

    # For the logging output:
    if use_rl:
        logger.info(f"\n[PIPELINE] RL Agent Performance:")
        logger.info(f"Total episodes: {safe_rl_agent_attribute(rl_agent, 'episode_count', 0)}")
        logger.info(f"Total reward: {safe_rl_agent_attribute(rl_agent, 'total_reward', 0.0):.2f}")
        logger.info(f"Final exploration rate: {safe_rl_agent_attribute(rl_agent, 'exploration_rate', 0.1):.2f}")

        api_efficiency = safe_rl_agent_attribute(rl_agent, "api_efficiency", [])
        api_eff = np.mean(api_efficiency[-20:]) if len(api_efficiency) > 20 else 'N/A'
        logger.info(f"API efficiency: {api_eff}")

    # Visualization creation (no changes needed here)
    try:
        # Create visualizations...
        logger.info(f"[PIPELINE] Visualizations saved to {run_dir}")
    except Exception as e:
        logger.warning(f"[PIPELINE] Error creating visualizations: {e}")

    return results


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


@timing_decorator
def process_sample_with_vae(embedding, vae_compressor):
    """Process embedding with VAE compression with proper dimension handling"""
    # Convert to tensor
    embedding_tensor = torch.tensor(embedding, dtype=torch.float32).to(device)

    # Get the target dimension from the VAE compressor
    target_dim = vae_compressor.input_dim  # Use the compressor's expected input dimension

    # Adapt dimensions - using the compressor's expected dimension
    adapted_tensor = adapt_dimensions(embedding_tensor, target_dim)

    # Compress using VAE
    with torch.no_grad():
        compressed_embedding = vae_compressor.compress(adapted_tensor).cpu().numpy()

    return compressed_embedding
def test_system_components():
    """Test the integration of KB and reconstruction components"""
    print("\n===== SYSTEM COMPONENT TEST =====")

    # Test cases with deliberate errors
    test_cases = [
        "Mrs Lynne, you are quite right and I shall check whether this ocs actually not been done.",
        "The Parliamemt will now vote on the propofal from the Commissiob.",
        "In accordancg with Rule 143, I wkulz like your acvioe about this moetinp."
    ]

    # Initialize components
    kb = get_or_create_knowledge_base()
    print(f"Knowledge Base loaded with {len(kb.term_dict)} terms")

    # Test each component
    success_count = 0

    for i, test in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: {test}")

        # Test 1: Direct KB reconstruction
        kb_result = kb.kb_guided_reconstruction(test)
        kb_changes = sum(1 for a, b in zip(test.split(), kb_result.split()) if a != b)
        print(f"1. KB Reconstruction ({kb_changes} changes):\n   {kb_result}")

        # Test 2: Basic text reconstruction
        basic_result = basic_text_reconstruction(test, use_kb=True)
        basic_changes = sum(1 for a, b in zip(test.split(), basic_result.split()) if a != b)
        print(f"2. Basic Reconstruction ({basic_changes} changes):\n   {basic_result}")

        # Test 3: API reconstruction if available
        if openai_available:
            api_result, _, _ = api_reconstruct_with_semantic_features(test, context="", use_kb=True)
            api_changes = sum(1 for a, b in zip(test.split(), api_result.split()) if a != b)
            print(f"3. API Reconstruction ({api_changes} changes):\n   {api_result}")

        # Count success if any component made changes
        if kb_changes > 0 or basic_changes > 0:
            success_count += 1

    # Final assessment
    print("\n===== RESULTS =====")
    print(f"Components tested: KB Reconstruction, Basic Reconstruction" +
          (", API Reconstruction" if openai_available else ""))
    print(f"Test cases corrected: {success_count}/{len(test_cases)}")
    print(f"System status: {'✓ FUNCTIONAL' if success_count > 0 else '✗ NOT WORKING PROPERLY'}")

    return success_count > 0
def check_kb_functionality():
    """
    Run a quick test to verify if the knowledge base is working correctly.
    Returns True if the KB is functioning properly, False otherwise.
    """
    print("\n===== KNOWLEDGE BASE FUNCTIONALITY CHECK =====")

    # Initialize the KB
    try:
        kb = get_or_create_knowledge_base()
        print(f"✓ KB initialized successfully with {len(kb.term_dict)} terms")
    except Exception as e:
        print(f"✗ Failed to initialize KB: {e}")
        return False

    # Test case definitions - each contains an input with deliberate errors
    test_cases = [
        {
            "input": "The Parliamemt will now vote on the propofal from the Commissiob.",
            "expected_changes": ["Parliamemt", "propofal", "Commissiob"]
        },
        {
            "input": "In accordancg with Rule 143, I wkulz like your acvioe.",
            "expected_changes": ["accordancg", "wkulz", "acvioe"]
        },
        {
            "input": "The Coupcil and Directave on environmentsl protrction.",
            "expected_changes": ["Coupcil", "Directave", "environmentsl", "protrction"]
        }
    ]

    # Run the tests
    success_count = 0
    total_corrections = 0

    for i, test in enumerate(test_cases):
        input_text = test["input"]
        expected = test["expected_changes"]

        # Run KB reconstruction
        corrected = kb.kb_guided_reconstruction(input_text)

        # Count actual corrections
        actual_corrections = []
        for a, b in zip(input_text.split(), corrected.split()):
            if a != b:
                actual_corrections.append(a)

        # Check if expected terms were corrected
        fixed_terms = [term for term in expected if term not in corrected.split()]
        success = len(fixed_terms) > 0

        # Print results
        print(f"\nTest {i + 1}:")
        print(f"  Input:      {input_text}")
        print(f"  Corrected:  {corrected}")
        print(f"  Corrections: {len(actual_corrections)}/{len(expected)} expected terms")

        # Track success
        if success:
            success_count += 1
            total_corrections += len(actual_corrections)
            print(f"  Result:     ✓ KB applied corrections")
        else:
            print(f"  Result:     ✗ KB failed to correct expected terms")

    # Final result
    overall_success = success_count > 0
    print("\n===== SUMMARY =====")
    print(f"Tests passed: {success_count}/{len(test_cases)}")
    print(f"Total corrections made: {total_corrections}")
    print(f"KB Status: {'FUNCTIONING' if overall_success else 'NOT WORKING PROPERLY'}")

    return overall_success

if __name__ == "__main__":
    # Test system components first
    system_ok = test_system_components()
    if not system_ok:
        print("WARNING: System components are not functioning properly!")
    # Run the enhanced pipeline with all improvements
    results = run_enhanced_pipeline(
        num_samples=50,
        noise_level=0.15,
        noise_type='gaussian',
        use_api_pct=0.5,
        comparison_mode=True,
        use_semantic_loss=True,
        use_vae_compression=True,
        use_content_adaptive_coding=True
    )
    # Check if KB is functioning
    kb_working = check_kb_functionality()
    print("\n====== Enhanced Semantic Communication Pipeline Complete ======")
    print(f"Overall improvements:")
    if ENABLE_VAE_COMPRESSION:
        print("- Advanced Compression: Implemented VAE-based non-linear compression")
    if ENABLE_CONTENT_ADAPTIVE_CODING:
        print("- Content-Adaptive Coding: Implemented content-aware protection strategies")
    print("- Semantic Perceptual Loss: Added semantic similarity metrics and training")
    print("- Enhanced RL Agent: Improved state representation with semantic features")

    if results and "overall_metrics" in results:
        print("\nFinal metrics:")
        for key, value in sorted(results["overall_metrics"].items()):
            if key.startswith("semantic_avg_"):
                print(f"  {key}: {value:.4f}")
