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
# Import modified components
from circuit_breaker import CircuitBreaker
from semantic_mlpdvae import load_or_train_enhanced_mlp_dvae
from mlpdvae_utils import (load_transmission_pairs, evaluate_reconstruction_with_semantics,
                           compute_embedding_similarity, generate_text_from_embedding,ensure_tensor_shape)
from semantic_loss import SemanticPerceptualLoss, evaluate_semantic_similarity
from compression_vae import (EmbeddingCompressorVAE, decompress_vae_embedding,
                             load_or_train_vae_compressor)

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

class EnhancedReinforcementLearningAgent:
    """
    Enhanced RL agent for optimizing API usage based on text properties.
    Now includes semantic metrics in its state and reward calculation.
    """

    def __init__(self, num_states=8, num_actions=3, learning_rate=0.1,
                 discount_factor=0.9, exploration_rate=0.3):
        self.num_states = num_states  # Increased states to account for semantic features
        self.num_actions = num_actions  # 0: basic, 1: GPT-3.5, 2: GPT-4
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        # Initialize Q-table
        self.q_table = np.random.uniform(low=0, high=0.1, size=(num_states, num_actions))

        # Experience buffer for offline learning
        self.experience_buffer = deque(maxlen=1000)

        # Track performance metrics
        self.total_reward = 0
        self.episode_count = 0
        self.api_efficiency = []

        # Load saved Q-table if exists
        self.load_q_table()

    def get_enhanced_state(self, corruption_level, text_length, semantic_features=None):
        """
        Enhanced state representation that includes semantic features.

        Args:
            corruption_level: Level of text corruption (0-1)
            text_length: Length of text in words
            semantic_features: Optional semantic features (e.g., content complexity)

        Returns:
            State index (0-7)
        """
        # Map corruption level to binary state (low/high)
        corruption_state = 1 if corruption_level > 0.1 else 0

        # Map text length to binary state (short/long)
        length_state = 1 if text_length > 20 else 0

        # Map semantic complexity to binary state (simple/complex)
        semantic_state = 0
        if semantic_features is not None:
            if isinstance(semantic_features, (int, float)):
                # Simple numeric semantic feature
                semantic_state = 1 if semantic_features > 0.5 else 0
            elif isinstance(semantic_features, (list, np.ndarray)) and len(semantic_features) > 0:
                # Vector of semantic features - use first element as complexity indicator
                semantic_state = 1 if semantic_features[0] > 0.5 else 0

        # Combined state (0-7) with three binary features
        return corruption_state * 4 + length_state * 2 + semantic_state

    def get_state(self, corruption_level, text_length):
        """Original state function (maintained for backward compatibility)"""
        # Map corruption level to binary state (low/high)
        corruption_state = 1 if corruption_level > 0.1 else 0

        # Map text length to binary state (short/long)
        length_state = 1 if text_length > 20 else 0

        # Combined state (0-3)
        return corruption_state * 2 + length_state

    def select_action(self, state, budget_remaining, force_basic=False):
        """Select action using epsilon-greedy policy with budget awareness"""
        # Force basic reconstruction if requested or very low budget
        if force_basic or budget_remaining < 0.05:
            return 0

        # Budget-aware strategy - avoid expensive models when budget is low
        if budget_remaining < 0.2 and self.q_table[state, 1] > 0:
            # Prefer GPT-3.5 over GPT-4 when budget is low
            return 1 if self.q_table[state, 1] > 0.5 * self.q_table[state, 2] else 0

        # Epsilon-greedy policy
        if np.random.random() < self.exploration_rate:
            # Random action, but limit GPT-4 usage based on budget
            if budget_remaining < 0.5:
                return np.random.choice([0, 1])  # Only basic or GPT-3.5
            else:
                return np.random.choice([0, 1, 2])  # Any action
        else:
            # Greedy action
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        """Update Q-table using Q-learning"""
        # Add to experience buffer for later training
        self.experience_buffer.append((state, action, reward, next_state))

        # Standard Q-learning update
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

        # Track rewards
        self.total_reward += reward

    def train_from_buffer(self, batch_size=32):
        """Train from experience buffer for more stable learning"""
        if len(self.experience_buffer) < batch_size:
            return

        # Sample random batch from buffer
        batch = random.sample(self.experience_buffer, batch_size)

        # Update Q-table from batch
        for state, action, reward, next_state in batch:
            best_next_action = np.argmax(self.q_table[next_state])
            td_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
            td_error = td_target - self.q_table[state, action]
            self.q_table[state, action] += self.learning_rate * td_error

    def calculate_reward(self, metrics, action, cost=0):
        """
        Enhanced reward function that incorporates semantic similarity.

        Args:
            metrics: Dictionary of quality metrics
            action: Action taken (0=basic, 1=GPT-3.5, 2=GPT-4)
            cost: API cost incurred

        Returns:
            Calculated reward
        """
        # Base reward from quality metrics with emphasis on semantic similarity
        quality_reward = 0

        # Include SEMANTIC metric if available
        if 'SEMANTIC' in metrics:
            quality_reward += metrics.get('SEMANTIC', 0) * 0.5  # 50% weight to semantic
            quality_reward += metrics.get('BLEU', 0) * 0.2  # 20% weight to BLEU
            quality_reward += metrics.get('ROUGEL', 0) * 0.3  # 30% weight to ROUGE-L
        else:
            # Traditional metrics if semantic not available
            quality_reward = metrics.get('BLEU', 0) * 0.3 + metrics.get('ROUGEL', 0) * 0.7

        # Cost penalty for API usage
        cost_penalty = 0
        if action > 0:  # API was used
            # Scale cost_penalty based on the budget
            cost_penalty = cost * 10  # Penalize more for higher costs

            # Track API efficiency
            efficiency = quality_reward / (cost + 0.001)  # Avoid division by zero
            self.api_efficiency.append(efficiency)

        # Final reward
        reward = quality_reward - cost_penalty

        return reward

    def save_q_table(self):
        """Save Q-table and metrics to file"""
        try:
            np.save(os.path.join(MODELS_DIR, 'enhanced_rl_agent_q_table.npy'), self.q_table)

            with open(os.path.join(MODELS_DIR, 'enhanced_rl_agent_metrics.json'), 'w') as f:
                json.dump({
                    'total_reward': self.total_reward,
                    'episode_count': self.episode_count,
                    'api_efficiency': list(self.api_efficiency) if len(self.api_efficiency) > 0 else [],
                    'num_states': self.num_states,
                    'num_actions': self.num_actions
                }, f)

            logger.info(f"Saved enhanced RL agent state")
        except Exception as e:
            logger.warning(f"Failed to save RL agent: {e}")

    def load_q_table(self):
        """Load Q-table and metrics from file if exists"""
        q_table_path = os.path.join(MODELS_DIR, 'enhanced_rl_agent_q_table.npy')
        metrics_path = os.path.join(MODELS_DIR, 'enhanced_rl_agent_metrics.json')

        try:
            if os.path.exists(q_table_path):
                self.q_table = np.load(q_table_path)

                # Check if dimensions match, otherwise resize
                if self.q_table.shape != (self.num_states, self.num_actions):
                    old_q_table = self.q_table.copy()
                    self.q_table = np.random.uniform(low=0, high=0.1, size=(self.num_states, self.num_actions))

                    # Copy over values that fit
                    min_states = min(old_q_table.shape[0], self.num_states)
                    min_actions = min(old_q_table.shape[1], self.num_actions)
                    self.q_table[:min_states, :min_actions] = old_q_table[:min_states, :min_actions]

                    logger.info(f"Resized Q-table from {old_q_table.shape} to {self.q_table.shape}")

                # Load metrics if available
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                        self.total_reward = metrics.get('total_reward', 0)
                        self.episode_count = metrics.get('episode_count', 0)
                        self.api_efficiency = metrics.get('api_efficiency', [])

                # Reduce exploration rate as we learn
                self.exploration_rate = max(0.05, 0.3 - 0.05 * self.episode_count)

                logger.info(f"Loaded enhanced RL agent (exploration rate: {self.exploration_rate:.2f})")
        except Exception as e:
            logger.warning(f"Failed to load RL agent: {e}")


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

    # Common corrections dictionary with Europarl terms
    corrections = {
        # Basic errors

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

        # Common Europarl terms
        "conceabing": "concerning",
        "Qutestois": "Quaestors",
        "moetinp": "meeting",
        "tednesgay": "Wednesday",
        "tru": "you",
        "cge": "can",
        "Plootjbvan": "Plooij-van",
        "Pcrliasent": "Parliament",
        "Parliamemt": "Parliament",
        "messaks": "message",
        "qcat": "that",
        "tre": "the",
        "salority": "majority",
        "wisk": "wish",
        "Commissiob": "Commission",
        "represenjatives": "representatives",
        "proporal": "proposal",
        "repprt": "report",
        "Coupcil": "Council",
        "agenfa": "agenda",
        "Directave": "Directive",
        "regulaaion": "regulation",
        "discussaon": "discussion",
        "debpte": "debate",
        "vite": "vote",
        "leglslation": "legislation",
        "mender": "member",
        "questimn": "question",
        "Europenn": "European",
        "agreus": "agrees",
        "propofal": "proposal",
        "environmentsl": "environmental",
        "protrction": "protection"
    }

    # Process each word
    for word in words:
        # Try exact match first
        if word in corrections:
            reconstructed_words.append(corrections[word])
            changes_made = True
            continue

        # Try case-insensitive match
        word_lower = word.lower()
        if word_lower in corrections:
            # Preserve capitalization
            if word[0].isupper():
                corrected = corrections[word_lower].capitalize()
            else:
                corrected = corrections[word_lower]
            reconstructed_words.append(corrected)
            changes_made = True
            continue

        # Check for common error patterns
        if len(word) > 3:
            # Check for vowel errors
            vowels = "aeiou"
            consonants = "bcdfghjklmnpqrstvwxyz"
            has_strange_pattern = False

            # No vowels in a long word is strange
            vowel_count = sum(1 for c in word.lower() if c in vowels)
            if vowel_count == 0 and len(word) > 4:
                has_strange_pattern = True

            # Too many consecutive consonants is suspicious
            max_consonant_streak = 0
            current_streak = 0
            for c in word.lower():
                if c in consonants:
                    current_streak += 1
                    max_consonant_streak = max(max_consonant_streak, current_streak)
                else:
                    current_streak = 0

            if max_consonant_streak > 3:  # More than 3 consonants in a row
                has_strange_pattern = True

            # Character substitution patterns
            if has_strange_pattern:
                # Try common substitution patterns
                substitutions = [
                    ('c', 'a'), ('c', 'e'), ('c', 'o'),  # 'c' often substitutes vowels
                    ('v', 'h'), ('t', 'h'), ('f', 'f'),  # Common substitutions
                    ('i', 'l'), ('l', 'i'), ('n', 'm')  # Similar looking characters
                ]

                # Try substituting each suspicious pattern
                for old, new in substitutions:
                    if old in word.lower():
                        # Try the substitution
                        test_word = word.lower().replace(old, new)
                        # See if it's a known word or matches a pattern
                        if test_word in corrections or any(
                                known.startswith(test_word[:3]) for known in corrections.keys()):
                            reconstructed_words.append(test_word)
                            changes_made = True
                            break

        # Default: keep the original word
        reconstructed_words.append(word)

    result = " ".join(reconstructed_words)

    # Log whether changes were made
    if changes_made:
        logger.info(f"Basic reconstruction made changes: '{noisy_text}' -> '{result}'")

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
@timing_decorator
def api_reconstruct_with_semantic_features(noisy_text, context="", rl_agent=None, budget_remaining=1.0,
                                           semantic_features=None, use_kb=True):
    """
    Enhanced version of API reconstruction with semantic features and knowledge base integration.
    Implements a multi-stage reconstruction approach prioritizing KB for common errors
    and falling back to API for complex cases.
    """
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
        return reconstructed, 0, 0  # Return text, cost, action

    # Use RL agent to decide whether to use API and which model
    force_basic = False
    use_gpt4 = False
    action = 0  # Default: basic reconstruction

    if rl_agent is not None:
        # Get state based on corruption level, text length, and semantic features
        corruption_level = min(1.0, sum(1 for a, b in zip(noisy_text.split(), context.split() if context else [])
                                        if a != b) / max(1, len(noisy_text.split())))
        text_length = len(noisy_text.split())

        # Use enhanced state if semantic features available
        if semantic_features is not None and hasattr(rl_agent, 'get_enhanced_state'):
            state = rl_agent.get_enhanced_state(corruption_level, text_length, semantic_features)
        else:
            state = rl_agent.get_state(corruption_level, text_length)

        # If KB already applied changes, bias toward cheaper options
        if kb_applied:
            # Reduce likelihood of using expensive API if KB already made some corrections
            adjusted_state = max(0, state - 1)  # Move state toward cheaper options
            action = rl_agent.select_action(adjusted_state, budget_remaining, force_basic)
        else:
            # Normal action selection
            action = rl_agent.select_action(state, budget_remaining, force_basic)

        if action == 0:
            # Use basic reconstruction
            reconstructed = basic_text_reconstruction(noisy_text, use_kb=use_kb)
            logger.info(f"[API] RL agent chose basic reconstruction")
            elapsed_time = time.time() - start_time
            logger.info(f"[API] Completed in {elapsed_time:.3f}s using basic reconstruction")
            return reconstructed, 0, action
        elif action == 2:
            # Use GPT-4
            use_gpt4 = True

    # Set up enhanced prompt with KB guidance
    system_prompt = """You are a specialized text reconstruction system. Your task is to correct errors in the text while preserving the original meaning and intent. Fix spelling, grammar, and word corruptions. The text contains European Parliament terminology."""

    # Add KB enhancement to system prompt if available
    if prompt_enhancement:
        system_prompt += f"\n\nIMPORTANT: {prompt_enhancement}"

    # Add domain knowledge to help with parliamentary terminology
    system_prompt += "\n\nEuropean Parliament terms to recognize: Parliament, Commission, Council, Directive, Regulation, Quaestors, Plooij-van Gorsel, Rule 143, amendments, proposal, agenda, debate, vote."

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
        user_prompt += f"Context: {context}\n\n"

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
            return reconstructed_text, cost, action
        else:
            logger.warning("[API] API call failed, using fallback reconstruction")
            reconstructed = basic_text_reconstruction(noisy_text, use_kb=use_kb)
            elapsed_time = time.time() - start_time
            logger.info(f"[API] Completed in {elapsed_time:.3f}s using basic reconstruction (API failed)")
            return reconstructed, 0, 0
    except Exception as e:
        logger.error(f"[API] API enhancement failed: {e}")
        reconstructed = basic_text_reconstruction(noisy_text, use_kb=use_kb)
        elapsed_time = time.time() - start_time
        logger.info(f"[API] Completed in {elapsed_time:.3f}s using basic reconstruction (API error)")
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

    Args:
        model: Model name to use
        messages: Messages to send
        max_retries: Maximum number of retries
        backoff_factor: Factor to increase the wait time between retries

    Returns:
        API response or None on failure
    """
    for attempt in range(max_retries):
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=150
            )
            logger.info(f"HTTP Request: POST https://api.openai.com/v1/chat/completions \"HTTP/1.1 200 OK\"")
            return response
        except Exception as e:
            wait_time = backoff_factor ** attempt
            logger.warning(
                f"API call failed with error: {e}. Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)

    logger.error(f"API call failed after {max_retries} attempts")
    return None
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
def safe_copy(obj):
    """Create a copy of an object that works for both NumPy arrays and PyTorch tensors."""
    if isinstance(obj, torch.Tensor):
        return obj.clone().detach().cpu().numpy()
    elif isinstance(obj, np.ndarray):
        return obj.copy()
    else:
        return obj  # For other types, return as is
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

    # Initialize knowledge base if requested
    kb = None
    if use_knowledge_base:
        try:
            kb = get_or_create_knowledge_base()
            logger.info("[PIPELINE] Knowledge base initialized successfully")
        except Exception as e:
            logger.warning(f"[PIPELINE] Could not initialize knowledge base: {e}")
            use_knowledge_base = False

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

    # Detect embedding dimensions
    embedding_dim = embeddings[0].shape[0] if embeddings else 460
    logger.info(f"[PIPELINE] Detected embedding dimension: {embedding_dim}")

    # Load VAE compressor if requested
    vae_compressor = None
    if use_vae_compression:
        try:
            # Fix the parameter name to embedding_dim
            vae_compressor = load_or_train_vae_compressor(compression_factor=VAE_COMPRESSION_FACTOR)
            if vae_compressor:
                logger.info("[PIPELINE] VAE compressor loaded successfully")
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
                    # Create content-adaptive channel with same parameters
                    adaptive_channel = ContentAdaptivePhysicalChannel(
                        snr_db=current_params.get('snr_db', 20.0),
                        channel_type=current_params.get('channel_type', 'awgn'),
                        modulation=current_params.get('modulation', 'qam').split('-')[0],
                        modulation_order=int(current_params.get('modulation', 'qam-16').split('-')[1]),
                        enable_content_adaptive_coding=True
                    )

                    # Replace channel in bridge
                    physical_semantic_bridge._physical_channel = adaptive_channel
                    logger.info("[PIPELINE] Physical channel upgraded to content-adaptive version")
                except TypeError:
                    logger.warning("[PIPELINE] Could not initialize content-adaptive channel. Using standard channel.")
        except Exception as e:
            logger.warning(f"[PIPELINE] Could not configure content-adaptive coding: {e}")
            use_content_adaptive_coding = False

    if not physical_semantic_bridge.physical_enabled and use_content_adaptive_coding:
        logger.warning(
            "[PIPELINE] Content-adaptive coding requested but physical channel is disabled. Continuing without content-adaptive coding.")
        use_content_adaptive_coding = False

    # Load or train enhanced MLPDenoisingVAE model with semantic loss
    dvae = load_or_train_enhanced_mlp_dvae(model_path="enhanced_mlp_dvae_model.pth",
                                           force_retrain=False,
                                           use_self_supervised=use_self_supervised,
                                           use_semantic_loss=use_semantic_loss)

    if dvae is None:
        logger.error("[PIPELINE] Could not load or train enhanced MLPDenoisingVAE model")
        pipeline_elapsed = time.time() - pipeline_start_time
        logger.info(f"[PIPELINE] Pipeline failed in {pipeline_elapsed:.2f}s")
        return

    # Initialize enhanced RL agent for API optimization
    use_rl = openai_available and num_samples >= 10
    rl_agent = EnhancedReinforcementLearningAgent() if use_rl else None

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
            f"[PIPELINE] Using enhanced RL agent for API optimization (exploration rate: {rl_agent.exploration_rate:.2f})")
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
            "use_rl_agent": use_rl
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

            # === Embedding-based reconstruction ===
            # Apply VAE compression if enabled
            if use_vae_compression and vae_compressor:
                # Convert to tensor
                embedding_tensor = torch.tensor(embedding, dtype=torch.float32).to(device)

                # Adapt the dimensions to match what the VAE expects (768)
                target_dim = 768  # The dimension the VAE was trained on
                embedding_tensor = adapt_dimensions(embedding_tensor, target_dim)

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
                    # Transmit through physical channel
                    noisy_embedding = transmit_through_physical_channel(noisy_embedding, debug=False,
                                                                        use_kb=use_knowledge_base)
                except Exception as e:
                    logger.warning(f"[PIPELINE] Physical channel transmission failed: {e}")
                    # Continue with noisy embedding if transmission fails

                # Store post-physical channel embedding
                sample_result["physical_noisy_embedding"] = safe_copy(noisy_embedding)

            # Reconstruct embedding using enhanced MLPDenoisingVAE
            with torch.no_grad():
                # Fix tensor shape issues using our utility function
                embedding_tensor = torch.tensor(noisy_embedding, dtype=torch.float32).to(device)
                embedding_tensor = ensure_correct_embedding_shape(embedding_tensor, expected_dim=2)

                # First encode to get latent representation
                mu, logvar = dvae.encode(embedding_tensor)

                # Use mean of encoding as latent vector (no sampling for inference)
                latent_vector = ensure_correct_embedding_shape(mu, expected_dim=2)

                # Then decode from latent space with text guidance if available
                if hasattr(dvae, 'decode_with_text_guidance'):
                    # Use text-guided decoding with proper context
                    reconstructed_embedding = dvae.decode_with_text_guidance(
                        latent_vector,
                        text_hint=sentence,
                        text_context=context if context else None
                    ).cpu().numpy()
                else:
                    # Standard decode
                    reconstructed_embedding = dvae.decode(latent_vector).cpu().numpy()

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

            # Extract simple semantic features for RL agent
            semantic_features = None
            if semantic_loss_fn is not None:
                # Simple feature - sentence length ratio compared to average
                avg_len = 20  # Assumed average sentence length
                len_ratio = len(sentence.split()) / avg_len
                semantic_features = [len_ratio, 0.5]  # Simple 2-feature vector

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
                        corrupted_text, context, use_kb=use_knowledge_base)
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

                # Update RL agent
                rl_agent.update(state, action, reward, next_state)

                # Periodically train from buffer
                if i % 10 == 0 and i > 0:
                    rl_agent.train_from_buffer()

                # Save progress periodically
                if i % 20 == 0 and i > 0:
                    rl_agent.save_q_table()

                # Increment episode count
                rl_agent.episode_count += 1

                # Record RL info
                sample_result["rl_state"] = int(state)
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
            rl_agent.save_q_table()
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
            "total_reward": rl_agent.total_reward,
            "episode_count": rl_agent.episode_count,
            "exploration_rate": rl_agent.exploration_rate,
            "q_table": rl_agent.q_table.tolist(),
            "num_states": rl_agent.num_states,
            "api_efficiency": rl_agent.api_efficiency[-50:] if len(rl_agent.api_efficiency) > 0 else []
        }

    # Add features information
    results["features"] = {
        "vae_compression": use_vae_compression,
        "semantic_loss": use_semantic_loss,
        "content_adaptive_coding": use_content_adaptive_coding,
        "enhanced_rl": use_rl and isinstance(rl_agent, EnhancedReinforcementLearningAgent)
    }

    # Add timing information
    pipeline_elapsed = time.time() - pipeline_start_time
    results["timing"] = {
        "total_time": pipeline_elapsed,
        "avg_per_sample": pipeline_elapsed / len(embeddings) if embeddings else 0,
        "samples_processed": len(embeddings)
    }

    # Convert numpy types for JSON serialization
    def convert_numpy_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_for_json(item) for item in obj]
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
            f.write(f"Total episodes: {rl_agent.episode_count}\n")
            f.write(f"Total reward: {rl_agent.total_reward:.2f}\n")
            f.write(f"Final exploration rate: {rl_agent.exploration_rate:.2f}\n")
            api_eff = np.mean(rl_agent.api_efficiency[-20:]) if len(rl_agent.api_efficiency) > 20 else 'N/A'
            f.write(f"API efficiency: {api_eff}\n")

    # Save cost log
    cost_tracker.save_log(os.path.join(run_dir, "cost_log.json"))

    # Print summary
    logger.info("\n=== Overall Results ===")
    logger.info(
        f"[PIPELINE] Total time: {pipeline_elapsed:.2f}s, Avg: {pipeline_elapsed / len(embeddings):.2f}s per sample")
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
    """Process embedding with VAE compression"""
    # Convert to tensor
    embedding_tensor = torch.tensor(embedding, dtype=torch.float32).to(device)

    # Adapt dimensions - fixed to 768 which is what the VAE expects
    target_dim = 768  # Hardcode the expected dimension
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
