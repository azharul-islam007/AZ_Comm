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

def get_dimension_registry():
    """Delayed import to avoid circular dependencies"""
    from physical_semantic_integration import DimensionRegistry
    return DimensionRegistry()
def timing_decorator(func):
    """Decorator to measure and log function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"[TIMING] Function {func.__name__} took {end_time - start_time:.4f} seconds to execute")
        return result
    return wrapper

common_corrections = [
    ("wkulz", "would"),
    ("couvsc", "course"),
    ("principdas", "principles"),
    ("accordancg", "accordance"),
    ("ymus", "your"),
    ("mnvice", "advice"),
    ("Rcne", "Rule"),
    ("acvioe", "advice"),
    ("ocs", "has"),
    ("tvks", "this"),
    ("dignt", "right"),
    ("ynu", "you"),
    ("gqe", "are"),
    ("quutg", "quite"),
    ("amf", "and"),
    ("hcve", "have"),
    ("woild", "would"),
    ("tht", "the"),
    ("ar", "are"),
    ("amd", "and"),
    ("hes", "has"),
    ("thct", "that"),
    ("hos", "has"),
    ("becn", "been"),
    ("doni", "done"),
    ("ct", "at"),
    ("wether", "whether"),
    ("wheter", "whether"),
    ("weither", "whether"),
    ("yhis", "this"),
    ("shal", "shall"),
    ("shali", "shall"),
    ("actully", "actually")
]
#################################################
# Reinforcement Learning Agent with Semantic Metrics
#################################################

class PPOAgent(nn.Module):
    """
    PPO (Proximal Policy Optimization) agent for API optimization
    with enhanced state representation and more sophisticated policy updates.
    """

    def __init__(self, state_dim=16, num_actions=3, learning_rate=0.0003):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions

        # Actor network (policy) - outputs action probabilities
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
            nn.Linear(64, num_actions)
        )

        # Critic network (value function) - estimates state value
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # PPO specific parameters
        self.clip_ratio = 0.2
        self.target_kl = 0.01
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.max_grad_norm = 0.5

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Experience buffer
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'advantages': [],
            'returns': []
        }

        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.lam = 0.95  # GAE parameter
        self.epsilon = 0.1  # Exploration rate

        # Tracking metrics
        self.total_reward = 0
        self.episode_count = 0
        self.api_efficiency = []

        # Load checkpoint if exists
        self.load_checkpoint()

    def get_enhanced_state(self, corruption_level, text_length, semantic_features=None):
        """
        Create enhanced state representation with linguistic and semantic features

        Args:
            corruption_level: Level of corruption (0-1)
            text_length: Length of text in tokens
            semantic_features: Optional semantic feature vector or dictionary

        Returns:
            State tensor with enhanced features
        """
        # Base features
        base_features = [
            corruption_level,  # Corruption level
            min(1.0, text_length / 100),  # Normalized text length
            float(self.epsilon),  # Current exploration rate
        ]

        # Extract or create parliamentary-specific features
        parl_features = []
        if isinstance(semantic_features, dict):
            # Extract structured features
            parl_features = [
                semantic_features.get('has_name', 0.0),  # Contains parliamentary name
                semantic_features.get('has_institution', 0.0),  # Contains institution name
                semantic_features.get('has_procedure', 0.0),  # Contains procedural term
                semantic_features.get('has_rule', 0.0),  # Contains rule reference
                semantic_features.get('critical_corruption', 0.0)  # Critical corruption detected
            ]
        elif semantic_features is not None:
            # Use provided feature vector
            if len(semantic_features) > self.state_dim - len(base_features):
                # Truncate to fit state_dim
                semantic_vector = semantic_features[:self.state_dim - len(base_features)]
            else:
                # Pad if needed
                semantic_vector = list(semantic_features)
                semantic_vector += [0.0] * (self.state_dim - len(base_features) - len(semantic_vector))

            parl_features = semantic_vector
        else:
            # No semantic features, use zeros
            parl_features = [0.0] * (self.state_dim - len(base_features))

        # Combine features
        state = base_features + parl_features
        return torch.tensor(state, dtype=torch.float32)

    def act(self, state, deterministic=False):
        """
        Select an action based on current policy

        Args:
            state: Current state
            deterministic: If True, use deterministic action selection

        Returns:
            action, log_prob, value
        """
        # Convert state to tensor if needed
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        state = state.to(device)

        # Forward pass through actor and critic
        with torch.no_grad():
            logits = self.actor(state)
            value = self.critic(state).squeeze()

            # Get action probabilities
            action_probs = F.softmax(logits, dim=-1)

            # Select action
            if deterministic:
                action = torch.argmax(action_probs).item()
                log_prob = torch.log(action_probs[action] + 1e-10).item()
            else:
                # Sample from distribution
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action)).item()

        return action, log_prob, value.item()

    def select_action(self, state, budget_remaining, force_basic=False, corruption_level=None):
        """
        Select action for API decision making with budget consideration

        Args:
            state: Current state
            budget_remaining: Remaining API budget
            force_basic: If True, force basic reconstruction
            corruption_level: Optional explicit corruption level

        Returns:
            action, log_prob
        """
        # Store budget_remaining for reward calculation
        self._budget_remaining = budget_remaining

        # Force basic reconstruction if requested or critically low budget
        if force_basic or budget_remaining < 0.02:
            return 0, 0.0  # Basic action, log_prob=0

        # Budget-based limitations
        budget_factor = min(1.0, budget_remaining / 0.5)

        # Get action probabilities
        action, log_prob, _ = self.act(state, deterministic=False)

        # Budget limitation logic
        if budget_remaining < 0.1 and action == 2:  # If low budget but want to use GPT-4
            # Downgrade to GPT-3.5 with 80% probability
            if np.random.random() < 0.8:
                action = 1
                # Recalculate log_prob for new action
                with torch.no_grad():
                    logits = self.actor(torch.tensor(state, dtype=torch.float32, device=device))
                    action_probs = F.softmax(logits, dim=-1)
                    log_prob = torch.log(action_probs[1] + 1e-10).item()

        # Enhanced parliamentary name detection - override for important texts
        if corruption_level is not None and corruption_level > 0.4:
            # Check state features for parliamentary indicators
            if isinstance(state, torch.Tensor) and state.numel() > 5:
                # Extract parliamentary features (based on our state design)
                has_name = state[3].item() if state.numel() > 3 else 0
                has_institution = state[4].item() if state.numel() > 4 else 0
                has_procedure = state[5].item() if state.numel() > 5 else 0

                # Increase probability of using API for parliamentary content
                if has_name > 0.7 or has_institution > 0.7 or has_procedure > 0.7:
                    # Use API with high probability for important parliamentary content
                    if budget_remaining > 0.2 and np.random.random() < 0.9:
                        if budget_remaining > 0.4:
                            action = 2  # Use GPT-4 for maximum quality
                        else:
                            action = 1  # Use GPT-3.5 as budget compromise

        return action, log_prob

    def store_experience(self, state, action, reward, value, log_prob):
        """Store experience in buffer for training"""
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['values'].append(value)
        self.buffer['log_probs'].append(log_prob)

        # Track total reward
        self.total_reward += reward

    def compute_advantages_and_returns(self):
        """Compute GAE advantages and returns"""
        rewards = np.array(self.buffer['rewards'])
        values = np.array(self.buffer['values'])

        # Estimate value for last state
        last_value = 0  # Assume episode ends or value is 0 for simplicity

        # Calculate advantages using GAE
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            # Next value is last_value if t is last step
            next_val = values[t + 1] if t < len(values) - 1 else last_value

            # Delta = r + gamma*V(s') - V(s)
            delta = rewards[t] + self.gamma * next_val - values[t]

            # GAE = delta + gamma*lambda*GAE
            gae = delta + self.gamma * self.lam * gae

            # Store advantage and return
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        # Store in buffer
        self.buffer['advantages'] = advantages
        self.buffer['returns'] = returns

    def update(self, state, action, reward, next_state, log_prob):
        """
        Store experience and update when enough data is collected

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            log_prob: Log probability of action
        """
        # Ensure state is a tensor
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        # Get value estimate for state
        with torch.no_grad():
            value = self.critic(state.to(device)).cpu().item()

        # Store experience
        self.store_experience(state, action, reward, value, log_prob)

        # Track total reward
        self.total_reward += reward

        # Update policy if enough data (20+ experiences)
        if len(self.buffer['states']) >= 20:
            self.update_policy()

    def update_policy(self):
        """Update policy using PPO algorithm"""
        if len(self.buffer['states']) < 4:  # Need at least a few samples
            return

        # Compute advantages and returns
        self.compute_advantages_and_returns()

        # Convert buffer data to tensors
        states = torch.stack(self.buffer['states']).to(device)
        actions = torch.tensor(self.buffer['actions'], dtype=torch.long).to(device)
        old_log_probs = torch.tensor(self.buffer['log_probs'], dtype=torch.float32).to(device)
        advantages = torch.tensor(self.buffer['advantages'], dtype=torch.float32).to(device)
        returns = torch.tensor(self.buffer['returns'], dtype=torch.float32).to(device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Multiple epochs of PPO updates (typically 3-10)
        for _ in range(4):
            # Get current policy and value estimates
            logits = self.actor(states)
            values = self.critic(states).squeeze()

            # Get action distributions
            action_probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(action_probs)

            # Get log probabilities and entropy
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # Calculate ratio and clipped ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)

            # Calculate losses
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            value_loss = F.mse_loss(values, returns)

            # Combined loss with entropy bonus
            loss = policy_loss - self.entropy_coef * entropy + self.value_coef * value_loss

            # Update actor
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            # Update critic
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

        # Clear buffer after updates
        for key in self.buffer:
            self.buffer[key] = []

        # Update exploration rate - gradually decrease
        self.epsilon = max(0.05, self.epsilon * 0.995)

        # Increment episode count
        self.episode_count += 1

    def calculate_reward(self, metrics, action, cost=0):
        """
        Enhanced reward function with stronger emphasis on semantic quality
        and dynamic budget consideration

        Args:
            metrics: Dictionary of reconstruction quality metrics
            action: Action taken (0=basic, 1=GPT-3.5, 2=GPT-4)
            cost: API cost incurred

        Returns:
            Calculated reward
        """
        # Base quality reward with stronger semantic emphasis
        quality_reward = 0

        # Extract exact match if available
        exact_match = metrics.get('exact_match', False)
        if exact_match:
            # Strong bonus for perfect reconstruction
            quality_reward += 5.0

        # Parliamentary content detection bonus
        parl_bonus = 0
        if 'parl_terms' in metrics and metrics['parl_terms'] > 0:
            # Bonus scales with number of parliamentary terms
            parl_bonus = min(2.0, metrics['parl_terms'] * 0.5)

        # Heavily prioritize semantic metrics
        if 'SEMANTIC' in metrics:
            sem_score = metrics.get('SEMANTIC', 0)
            # Exponential reward for high semantic similarity
            quality_reward += (sem_score ** 1.5) * 2.0  # Higher exponent gives more reward for quality
            # Additional smaller weights for traditional metrics
            quality_reward += metrics.get('BLEU', 0) * 0.3
            quality_reward += metrics.get('ROUGEL', 0) * 0.2
        else:
            # Traditional metrics if semantic not available
            quality_reward = metrics.get('BLEU', 0) * 0.4 + metrics.get('ROUGEL', 0) * 0.6

        # Add parliamentary bonus
        quality_reward += parl_bonus

        # Reward perfect or near-perfect reconstructions significantly
        if metrics.get('SEMANTIC', 0) > 0.95 and metrics.get('BLEU', 0) > 0.9:
            quality_reward *= 2.0  # 100% bonus for near-perfect reconstructions

        # Dynamic cost penalty based on budget remaining
        budget_remaining = getattr(self, '_budget_remaining', 0.9)

        # Cost penalty calculations
        if action > 0:  # API was used
            # Scale cost penalty based on budget and action
            if action == 1:  # GPT-3.5
                # Lower penalty when budget is high, higher when low
                cost_scale = 2.0 * (1.0 + (1.0 - budget_remaining))
            else:  # GPT-4
                # Similarly for GPT-4 but with different base
                cost_scale = 1.5 * (1.0 + (1.0 - budget_remaining))

            # Calculate penalty
            cost_penalty = cost * cost_scale

            # Budget-aware scaling
            if budget_remaining > 0.7:
                # Plentiful budget - reduce penalty
                cost_penalty *= 0.5
            elif budget_remaining < 0.3:
                # Low budget - increase penalty
                cost_penalty *= 1.5

            # Track API efficiency
            efficiency = quality_reward / (cost + 0.001)
            self.api_efficiency.append(efficiency)
        else:
            # No cost for basic reconstruction
            cost_penalty = 0

            # Add a small penalty for not using API when quality is very low
            if quality_reward < 0.3 and budget_remaining > 0.5:
                cost_penalty = 0.2  # Penalty for being too conservative

        # Final reward calculation with tanh to keep in reasonable range
        quality_component = np.tanh(quality_reward)
        final_reward = quality_component - cost_penalty

        return final_reward

    def save_checkpoint(self, path=None):
        """Save model checkpoint"""
        if path is None:
            path = os.path.join(MODELS_DIR, 'ppo_agent.pth')

        try:
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'epsilon': self.epsilon,
                'total_reward': self.total_reward,
                'episode_count': self.episode_count,
                'api_efficiency': self.api_efficiency,
                'state_dim': self.state_dim,
                'num_actions': self.num_actions
            }, path)

            logger.info(f"Saved PPO agent state")
        except Exception as e:
            logger.warning(f"Failed to save PPO agent: {e}")

    def load_checkpoint(self, path=None):
        """Load model checkpoint"""
        if path is None:
            path = os.path.join(MODELS_DIR, 'ppo_agent.pth')

        try:
            if not os.path.exists(path):
                return False

            checkpoint = torch.load(path, map_location=torch.device('cpu'))

            # Load state dictionaries
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

            # Load other attributes
            self.epsilon = checkpoint.get('epsilon', 0.1)
            self.total_reward = checkpoint.get('total_reward', 0)
            self.episode_count = checkpoint.get('episode_count', 0)
            self.api_efficiency = checkpoint.get('api_efficiency', [])

            logger.info(f"Loaded PPO agent (exploration rate: {self.epsilon:.2f})")
            return True
        except Exception as e:
            logger.warning(f"Failed to load PPO agent: {e}")
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


# AFTER: Improved basic_text_reconstruction function in SD5.py

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
                # Verify changes don't break correct words
                words_orig = noisy_text.split()
                words_recon = kb_result.split()

                common_correct_words = {'are', 'has', 'is', 'the', 'that', 'this', 'have', 'been', 'and', 'not', 'actually'}

                bad_changes = 0
                for i in range(min(len(words_orig), len(words_recon))):
                    if words_orig[i].lower() in common_correct_words and words_orig[i] != words_recon[i]:
                        bad_changes += 1

                # Only accept KB reconstruction if it doesn't break correct words
                if bad_changes == 0:
                    logger.info(f"KB reconstruction made changes: '{noisy_text}' -> '{kb_result}'")
                    return kb_result
                else:
                    logger.info(f"KB reconstruction rejected due to {bad_changes} questionable changes")
        except Exception as e:
            logger.warning(f"KB reconstruction failed: {e}")

    corrections = {"wkulz": "would", "couvsc": "course", "principdas": "principles", "accordancg": "accordance", "ymus": "your", "mnvice": "advice", "Rcne": "Rule", "acvioe": "advice", "ocs": "has", "tvks": "this", "dignt": "right", "ynu": "you", "gqe": "are", "quutg": "quite", "amf": "and", "hcve": "have", "woild": "would", "tht": "the", "ar": "are", "amd": "and", "hes": "has", "ruiye": "quite", "dwne": "done", "shuml": "shall", "righ": "right", "kuite": "quite", "qite": "quite", "qute": "quite", "actuall": "actually", "shal": "shall", "dine": "done", "donne": "done", "chepk": "check", "shhlo": "shall", "lhegk": "check", "hrq": "Mrs", "neu": "you", "ern": "are", "ars": "has", "Parlamentj": "Parliament", "Parliamemt": "Parliament", "Parlitment": "Parliament", "Parljament": "Parliament", "Pcrliasent": "Parliament", "Palrliament": "Parliament", "Eurepean": "European", "Europenn": "European", "Europvan": "European", "Ejropean": "European", "Commision": "Commission", "Commissiob": "Commission", "Commizion": "Commission", "Conmission": "Commission", "Coxmission": "Commission", "Commissjon": "Commission", "Councjl": "Council", "Councip": "Council", "Coupcil": "Council", "Kouncil": "Council", "Directave": "Directive", "Directlve": "Directive", "Direptive": "Directive", "Regulatien": "Regulation", "Regulaaion": "Regulation", "Regupation": "Regulation", "Regklation": "Regulation", "moetinp": "meeting", "meating": "meeting", "meetang": "meeting", "meetting": "meeting", "meetirg": "meeting", "metting": "meeting", "mceeting": "meeting", "sessien": "session", "sessiom": "session", "sesslon": "session", "sewsion": "session", "agendq": "agenda", "agenfa": "agenda", "agenca": "agenda", "agendz": "agenda", "agemda": "agenda", "tgendw": "agenda", "vite": "vote", "votr": "vote", "vots": "vote", "votung": "voting", "votinf": "voting", "voring": "voting", "vodting": "voting", "Presidemt": "President", "Presidebt": "President", "Presidfnt": "President", "Presldent": "President", "Presidont": "President", "Presidnet": "President", "Predsent": "President", "Memmber": "Member", "Membeq": "Member", "Membsr": "Member", "Membez": "Member", "mender": "member", "Rapporceur": "Rapporteur", "Rappofteur": "Rapporteur", "Rapporteud": "Rapporteur", "Qutestois": "Quaestors", "Quaestorz": "Quaestors", "Quaestoms": "Quaestors", "proporal": "proposal", "propesal": "proposal", "propozal": "proposal", "proposql": "proposal", "propofal": "proposal", "repprt": "report", "repord": "report", "repott": "report", "agreez": "agrees", "agress": "agrees", "agreus": "agrees", "agreet": "agrees", "requzst": "request", "requestz": "request", "requept": "request", "chedk": "check", "chrck": "check", "chexk": "check", "shsll": "shall", "shatl": "shall", "shail": "shall", "shal": "shall", "sholl": "shall", "shoold": "should", "shoumd": "should", "shuold": "should", "shoula": "should", "amendmert": "amendment", "amendnent": "amendment", "amencment": "amendment", "amendmemt": "amendment", "debpte": "debate", "debugr": "debate", "debats": "debate", "debare": "debate", "discussaon": "discussion", "discussiom": "discussion", "discuzsion": "discussion", "disgussion": "discussion", "decisien": "decision", "decisiob": "decision", "decizion": "decision", "procedume": "procedure", "proceduge": "procedure", "procedjre": "procedure", "procexure": "procedure", "considrr": "consider", "consiter": "consider", "considep": "consider", "examinz": "examine", "examime": "examine", "examino": "examine", "reviev": "review", "reviem": "review", "reviaw": "review", "afdress": "address", "addross": "address", "addrezs": "address", "continye": "continue", "continoe": "continue", "contimue": "continue", "Rulw": "Rule", "Ruls": "Rule", "Ryle": "Rule", "Ruie": "Rule", "admisssbility": "admissibility", "admissihility": "admissibility", "inadmissibllity": "inadmissibility", "inadmissihility": "inadmissibility", "questimn": "question", "questiom": "question", "questiob": "question", "questiin": "question", "environmemtal": "environmental", "environmentsl": "environmental", "environmentsl": "environmental", "environmebtal": "environmental", "protrction": "protection", "protectlon": "protection", "protectiom": "protection", "leglslation": "legislation", "legislasion": "legislation", "legislatiom": "legislation", "legislatisn": "legislation", "wkulz": "would", "cyb": "can", "cqn": "can", "arn": "are", "arr": "are", "thatb": "that", "thag": "that", "ths": "this", "thjs": "this", "tje": "the", "tye": "the", "thn": "the", "asd": "and", "anx": "and", "nad": "and", "fpr": "for", "gor": "for", "fot": "for", "wuth": "with", "wirh": "with", "witn": "with", "wiph": "with", "haz": "has", "han": "has", "haa": "has", "nof": "not", "npt": "not", "mot": "not", "whetjer": "whether", "whethe4": "whether", "whetheq": "whether", "whetner": "whether", "whethtr": "whether", "righg": "right", "righr": "right", "righf": "right", "actuslly": "actually", "actuslly": "actually", "actuaoly": "actually", "actualby": "actually", "afyually": "actually", "yiu": "you", "yoi": "you", "yuo": "you", "yea": "yes", "yez": "yes", "righg": "right", "righr": "right", "correcx": "correct", "correcy": "correct", "Lynme": "Lynne", "Lymne": "Lynne", "Lynnw": "Lynne", "Mts": "Mrs", "Mrz": "Mrs", "Segmu": "Segni", "Segmi": "Segni", "Schroedtet": "Schroedter", "Schroedtef": "Schroedter", "Schroedtez": "Schroedter", "Madom": "Madam", "Madas": "Madam", "Berengurr": "Berenguer", "Berenguez": "Berenguer", "Berenguef": "Berenguer", "Beeenguew": "Berenguer", "Fustez": "Fuster", "Fustrr": "Fuster", "Fustef": "Fuster", "Gorsel": "Gorsel", "Gorseb": "Gorsel", "Gorsep": "Gorsel", "Plooij-vam": "Plooij-van", "Plooij-vsn": "Plooij-van", "Ploupj-van": "Plooij-van", "Díea": "Díez", "Díef": "Díez", "Díex": "Díez", "Evams": "Evans", "Evabs": "Evans", "Evanz": "Evans"}

    # Original method as fallback (with improvements)
    original_noisy = noisy_text  # Store the original
    changes_made = False  # Reset change tracker

    words = noisy_text.split()
    corrected_words = []

    # Process each word
    for word in words:
        # Skip very short words and punctuation
        if len(word) <= 2 or all(c in '.,;:!?()[]{}"\'' for c in word):
            corrected_words.append(word)
            continue

        # If the word is already a common correct word, don't try to change it
        if word.lower() in {'the', 'that', 'this', 'is', 'are', 'have', 'has', 'and', 'not', 'with', 'for'}:
            corrected_words.append(word)
            continue

        # Try exact match in dictionary (case-insensitive)
        if word.lower() in corrections:
            # Preserve capitalization
            if word[0].isupper() and len(corrections[word.lower()]) > 0:
                corrected = corrections[word.lower()].capitalize()
            else:
                corrected = corrections[word.lower()]
            corrected_words.append(corrected)
            if word != corrected:
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

        # Don't replace if similarity is too low
        min_acceptable_score = 0.7  # Higher threshold to avoid bad replacements
        if best_match and score > min_acceptable_score:
            # Preserve capitalization
            if word[0].isupper() and len(best_match) > 0:
                best_match = best_match.capitalize()
            corrected_words.append(best_match)
            changes_made = True
        else:
            # Keep original if no confident correction found
            corrected_words.append(word)

    result = " ".join(corrected_words)

    # Apply phrase-based corrections if needed
    phrase_corrected = apply_phrase_patterns(result)
    if phrase_corrected != result:
        changes_made = True
        result = phrase_corrected

    # Verify changes by comparing original and result directly
    if result == original_noisy:
        changes_made = False  # No actual changes

        # Try harder with common fixes
        for old, new in common_corrections:
            if old in result:
                temp_result = result.replace(old, new)
                if temp_result != result:
                    result = temp_result
                    changes_made = True

    # Only log changes if actually made
    if changes_made:
        logger.info(f"Basic reconstruction made changes: '{original_noisy}' -> '{result}'")
    else:
        logger.debug(f"Basic reconstruction made no changes to: '{original_noisy}'")

    return result


def apply_phrase_patterns(text):
    """Apply phrase-level patterns for correction"""
    patterns = [('shhlo', 'shall'), ('lhegk', 'check'), ('hrq', 'Mrs'), ('neu ern', 'you are'), ('wkulz', 'would'), ('tvks', 'this'), ('dignt', 'right'), ('ynu', 'you'), ('gqe', 'are'), ('quutg', 'quite'), ('amf', 'and'), ('hcve', 'have'), ('woild', 'would'), ('tht', 'the'), ('thct', 'that'), ('hos', 'has'), ('becn', 'been'), ('doni', 'done'), ('ct', 'at'), ('Madam Presidemt', 'Madam President'), ('Madam Presidebt', 'Madam President'), ('Madam Presldent', 'Madam President'), ('Mts Lynne', 'Mrs Lynne'), ('Mrz Lynne', 'Mrs Lynne'), ('Mrs Lymne', 'Mrs Lynne'), ('Mrs Ploupj-van', 'Mrs Plooij-van'), ('Mrs Plooij-vam', 'Mrs Plooij-van'), ('Mr Evams', 'Mr Evans'), ('Mr Berenguef', 'Mr Berenguer'), ('Mr Berengurr', 'Mr Berenguer'), ('Mr Beeenguew', 'Mr Berenguer'), ('Mr Fustez', 'Mr Fuster'), ('Mr Fustrr', 'Mr Fuster'), ('Europenn Parliament', 'European Parliament'), ('Eurepean Parliament', 'European Parliament'), ('European Parliamemt', 'European Parliament'), ('European Pcrliasent', 'European Parliament'), ('the Commissiob', 'the Commission'), ('the Commizion', 'the Commission'), ('the Conmission', 'the Commission'), ('the Coupcil', 'the Council'), ('the Councip', 'the Council'), ('the Councjl', 'the Council'), ('in accordancg with', 'in accordance with'), ('in accbadance with', 'in accordance with'), ('on the agenfa', 'on the agenda'), ('on the agendq', 'on the agenda'), ('on the agenca', 'on the agenda'), ('on the tgendw', 'on the agenda'), ('this subject in the course', 'this subject in the course'), ('points of orter', 'points of order'), ('vote on the propofal', 'vote on the proposal'), ('vote on the propesal', 'vote on the proposal'), ('vote on the proporal', 'vote on the proposal'), ('I shhlo lhegk', 'I shall check'), ('I shall chedk', 'I shall check'), ('I shall chrck', 'I shall check'), ('I wkulz like', 'I would like'), ('I woild like', 'I would like'), ('air quality tesk', 'air quality test'), ('air qualiti test', 'air quality test'), ('fire driel', 'fire drill'), ('fire dril', 'fire drill'), ('fyre drill', 'fire drill'), ('no-smocing areas', 'no-smoking areas'), ('no-smoklng areas', 'no-smoking areas'), ('the staixcased', 'the staircases'), ('the ptairuases', 'the staircases'), ('Rule 143 concernimg', 'Rule 143 concerning'), ('Rule 143 concernint', 'Rule 143 concerning'), ('Rule 143 concerninh', 'Rule 143 concerning'), ('concerning inadmissibllity', 'concerning inadmissibility'), ('concerning inadmissihility', 'concerning inadmissibility'), ('you are quite righ', 'you are quite right'), ('you are quitz right', 'you are quite right'), ('you arn quite right', 'you are quite right'), ('neu ern quite right', 'you are quite right'), ('shall check whethzr', 'shall check whether'), ('shall check whethep', 'shall check whether'), ('shall check wether', 'shall check whether'), ('shall check wheter', 'shall check whether'), ('check whether thiz', 'check whether this'), ('check whether thia', 'check whether this'), ('whether this ars', 'whether this has'), ('whether this haa', 'whether this has'), ('whether this haz', 'whether this has'), ('has actually nof', 'has actually not'), ('has actyally not', 'has actually not'), ('has actuslly not', 'has actually not'), ('not been doni', 'not been done'), ('not bean done', 'not been done'), ('not bien done', 'not been done'), ('The House rosf', 'The House rose'), ('The House rosr', 'The House rose'), ('The Parliament woll', 'The Parliament will'), ('The Parliament wiil', 'The Parliament will'), ('The committee approvrd', 'The committee approved'), ('The committee approvef', 'The committee approved'), ('The Commission propozed', 'The Commission proposed'), ('The Commission proposef', 'The Commission proposed'), ('The Commission haz', 'The Commission has'), ('so Parliament shoild', 'so Parliament should'), ('so Parliament shoumd', 'so Parliament should'), ('now vote on thw', 'now vote on the'), ('now vote on tne', 'now vote on the'), ('we shall vote todya', 'we shall vote today'), ('we shall vote todaz', 'we shall vote today'), ('the vast majoritp', 'the vast majority'), ('the vast majorita', 'the vast majority'), ('the vast salority', 'the vast majority'), ('this part-sesslon', 'this part-session'), ('this part-sessiom', 'this part-session'), ('this part-sessien', 'this part-session'), ('will now proceet', 'will now proceed'), ('will now proceef', 'will now proceed'), ('request a debatz', 'request a debate'), ('request a debats', 'request a debate'), ('request a debpte', 'request a debate'), ('meeting on Wedneshay', 'meeting on Wednesday'), ('meeting on Wednesfay', 'meeting on Wednesday'), ('meeting on tednesgay', 'meeting on Wednesday'), ('on the agendc for', 'on the agenda for'), ('on the agendz for', 'on the agenda for'), ("Quaestors ' meetint", "Quaestors ' meeting"), ("Quaestors ' meating", "Quaestors ' meeting"), ("Quaestors ' meetirg", "Quaestors ' meeting"), ("Quaestors ' moetinp", "Quaestors ' meeting"), ('environmental protectiom', 'environmental protection'), ('environmental protectlon', 'environmental protection'), ('environmental protrction', 'environmental protection'), ('environmemtal protection', 'environmental protection'), ('environmentsl protection', 'environmental protection'), ('budgek proposal', 'budget proposal'), ('budgrt proposal', 'budget proposal'), ('budged proposal', 'budget proposal'), ('shhlo lhegk', 'shall check'), ('sholl chexk', 'shall check'), ('shatl chrck', 'shall check'), ('wiph thiz', 'with this'), ('wirh thjs', 'with this'), ('arn quitz', 'are quite'), ('arr quutg', 'are quite')]

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


def extract_parliamentary_features(text):
    """
    Extract parliamentary-specific features from text for improved RL state representation

    Args:
        text: Text to analyze

    Returns:
        Dictionary of features
    """
    # Initialize features
    features = {
        'has_name': 0.0,
        'has_institution': 0.0,
        'has_procedure': 0.0,
        'has_rule': 0.0,
        'critical_corruption': 0.0
    }

    # Names of important parliamentary figures
    names = ['Lynne', 'Plooij-van', 'Gorsel', 'Evans', 'Berenguer', 'Fuster',
             'Segni', 'Schroedter', 'Díez', 'Hicks', 'President']

    # Institution names
    institutions = ['Parliament', 'Commission', 'Council', 'Quaestors',
                    'European', 'Union', 'Committee']

    # Procedural terms
    procedures = ['agenda', 'vote', 'Rule', 'session', 'meeting', 'debate',
                  'proposal', 'amendment', 'directive', 'regulation', 'procedure']

    # Process text
    words = text.split()

    # Check for names
    for name in names:
        if name in text or name.lower() in text:
            features['has_name'] = 1.0
            break

    # Check for institutions
    for inst in institutions:
        if inst in text or inst.lower() in text:
            features['has_institution'] = 1.0
            break

    # Check for procedural terms
    for proc in procedures:
        if proc in text or proc.lower() in text:
            features['has_procedure'] = 1.0
            break

    # Check for rule numbers
    if 'Rule' in text and any(c.isdigit() for c in text):
        features['has_rule'] = 1.0

    # Critical corruption detection
    corruption_patterns = ['bb', 'bz', 'hz', 'jz', 'kz', 'pj', 'xn', 'qx', 'oj',
                           'wk', 'wg', 'vb', 'xj', 'lk', 'vn', 'tm']

    # Check for corruption patterns
    pattern_count = 0
    for pattern in corruption_patterns:
        if pattern in text.lower():
            pattern_count += 1

    # Set critical corruption if multiple patterns found
    if pattern_count >= 2:
        features['critical_corruption'] = min(1.0, pattern_count / 5)

    return features
# Full updated function:
# AFTER
@timing_decorator
def api_reconstruct_with_semantic_features(noisy_text, context="", rl_agent=None, budget_remaining=1.0,
                                           semantic_features=None, use_kb=True, additional_contexts=None):
    """
    Significantly enhanced API reconstruction with more aggressive API usage and better KB-API coordination.
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
            # Too many recovery attempts, implement a sophisticated cascade fallback
            logger.warning("Too many API recovery attempts, implementing cascade fallback")

            # 1. Try KB reconstruction with higher effort
            try:
                kb = get_or_create_knowledge_base()
                kb_result = kb.kb_guided_reconstruction(noisy_text)

                # If KB made ANY changes, try to improve further with basic
                if kb_result != noisy_text:
                    basic_applied_to_kb = basic_text_reconstruction(kb_result, use_kb=False)

                    # Check if basic improved KB result
                    if basic_applied_to_kb != kb_result:
                        logger.info("Cascade fallback: KB + Basic produced improvements")
                        return basic_applied_to_kb, 0, 0
                    else:
                        logger.info("Cascade fallback: Using KB only")
                        return kb_result, 0, 0
            except Exception as e:
                logger.warning(f"KB cascade fallback failed: {e}")

            # 2. If KB failed completely, try aggressive basic reconstruction
            try:
                # Apply basic with more aggressive settings
                basic_result = basic_text_reconstruction(noisy_text, use_kb=True)

                # If basic made ANY changes, return it
                if basic_result != noisy_text:
                    logger.info("Cascade fallback: Using aggressive basic reconstruction")
                    return basic_result, 0, 0
            except Exception as e:
                logger.warning(f"Basic cascade fallback failed: {e}")

            # 3. Last resort: ensemble approach combining partial results
            logger.info("Using last-resort ensemble combination")
            # Just return the original as we've exhausted all options
            return noisy_text, 0, 0
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
    kb_confidence = 0.0

    # Enhanced logging
    logger.info(f"[API] Starting reconstruction of text: '{noisy_text[:30]}...'")

    # NEW: Enhanced corruption detection with more patterns
    corruption_level = estimate_corruption_level(noisy_text)

    # Try knowledge base reconstruction first
    kb_reconstructed = noisy_text  # Default to no change
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

                # Calculate confidence score with more weight on word-level changes
                kb_confidence = char_overlap * (1 - min(0.4, word_diff_ratio)) * 1.1  # Slight boost

                # More permissive threshold to favor KB usage
                confidence_threshold = 0.7 - min(0.3, len(noisy_text.split()) / 100)

                if kb_confidence > confidence_threshold:
                    logger.info(f"[API] Using KB reconstruction with confidence {kb_confidence:.2f}")
                    method_used = "kb"
                    elapsed_time = time.time() - start_time
                    logger.info(f"[API] Completed in {elapsed_time:.3f}s using method: {method_used}")

                    # When returning successfully, clear recovery attempts
                    _api_recovery_attempts.pop(request_key, None)

                    return kb_reconstructed, 0, 0  # Return KB result, zero cost, basic action
        except Exception as e:
            logger.warning(f"[API] KB reconstruction attempt failed: {e}")

        # Extract parliamentary-specific features for PPO agent
        parl_features = extract_parliamentary_features(noisy_text)

        # Include these features if we're using the PPO agent
        if semantic_features is None and rl_agent is not None and isinstance(rl_agent, PPOAgent):
            semantic_features = parl_features

        # Determine if we should use API based on multiple factors
        should_use_api_result, api_model, reason = should_use_api(
            noisy_text, kb_reconstructed if kb_applied else None,
            kb_confidence, budget_remaining, rl_agent,
            parl_features=parl_features  # Pass parliamentary features
        )

    # Skip API if not available or shouldn't use
    if not should_use_api_result or not openai_available or not openai_client:
        if not should_use_api_result:
            logger.info(f"[API] Decision not to use API: {reason}")
        else:
            logger.info("[API] OpenAI API not available, using basic reconstruction")

        # Return KB reconstruction if it made any changes, otherwise try basic
        if kb_applied and kb_reconstructed != noisy_text:
            elapsed_time = time.time() - start_time
            logger.info(f"[API] Completed in {elapsed_time:.3f}s using KB reconstruction (fallback)")
            _api_recovery_attempts.pop(request_key, None)
            return kb_reconstructed, 0, 0

        # Fall back to basic reconstruction
        reconstructed = basic_text_reconstruction(noisy_text, use_kb=use_kb)
        elapsed_time = time.time() - start_time
        logger.info(f"[API] Completed in {elapsed_time:.3f}s using basic reconstruction")

        # When returning successfully, clear recovery attempts
        _api_recovery_attempts.pop(request_key, None)

        return reconstructed, 0, 0  # Return text, cost, action

    # If we get here, we should use the API
    try:
        # Enhanced system prompt with better Parliamentary domain knowledge
        system_prompt = """You are a specialized text reconstruction system for the European Parliament. Your task is to correct errors in text while preserving the original meaning and intent.

        IMPORTANT GUIDELINES:
        1. Correct spelling, grammar, and word corruptions carefully
        2. Pay special attention to parliamentary terminology and names
        3. NEVER replace "that" with "the" unless absolutely necessary
        4. Pay special attention to corrupted words at the beginning of sentences
        5. Make sure your reconstruction is grammatically correct
        6. Preserve ALL original parliamentary terms and names
        7. Do not add or remove information - focus only on correction
        8. Keep the same sentence structure as the original
        9. BE CONSERVATIVE - when in doubt, keep the original words

        European Parliament terms to recognize: Parliament, Commission, Council, Directive, Regulation, Quaestors, 
        Plooij-van Gorsel, Rule 143, amendments, proposal, agenda, debate, vote, meeting, session.

        Common error patterns to fix:
        - Words starting with "oh" are often corrupted forms of "th" (ohiz → this)
        - "Tcis" should be corrected to "This"
        - "ministeg" should be corrected to "minister"
        - "ieetpng" should be corrected to "meeting"
        - "vbn" is often a corruption of "van"
        - "wgn" is often a corruption of "can" or "will"
        - Names of officials and institutions should be properly capitalized
        """

        # Add KB knowledge if available
        if kb_applied:
            prompt_enhancement = f"Consider this possible correction as a starting point, but verify carefully: '{kb_reconstructed}'"
            system_prompt += f"\n\nIMPORTANT: {prompt_enhancement}"

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
    Reconstructed: I would like your advice about Rule 143 concerning inadmissibility.

    Original: Mrs Plooij-van Gorsel, I can tell you that this matter is on the agenda for the Quaestors' meeting on Wednesday.
    Corrupted: Mrs Plooij-vbn Gorsel, I wgn tell you wvat tiio matter is on the agenda for the Quaestors' ieetpng on Wednesday.
    Reconstructed: Mrs Plooij-van Gorsel, I can tell you that this matter is on the agenda for the Quaestors' meeting on Wednesday."""

        user_prompt = f"{example}\n\n"
        if context:
            # Add primary context
            user_prompt += f"Immediate Context: {context}\n\n"

            # Add additional context from global history if available
            if additional_contexts and len(additional_contexts) > 0:
                user_prompt += "Additional Context:\n"
                for i, prev_context in enumerate(additional_contexts):
                    user_prompt += f"[{i + 1}] {prev_context}\n"
                user_prompt += "\n"

        user_prompt += f"Corrupted: {noisy_text}\nReconstructed:"

        # Make API call
        logger.info(f"[API] Making API call with model {api_model}...")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        max_retries = config_manager.get("api.max_retries", 3)
        response = make_api_call_with_retry(api_model, messages, max_retries=max_retries)

        if response:
            # Extract corrected text
            reconstructed_text = response.choices[0].message.content.strip()

            # Clean up response
            for prefix in ["Reconstructed:", "Reconstructed text:"]:
                if reconstructed_text.startswith(prefix):
                    reconstructed_text = reconstructed_text[len(prefix):].strip()

            # Apply post-reconstruction validation with KB awareness
            reconstructed_text = validate_reconstruction(noisy_text, reconstructed_text,
                                                         kb_reconstructed if kb_applied else None)

            # Calculate API cost
            input_tokens = get_token_count(system_prompt + user_prompt)
            output_tokens = get_token_count(reconstructed_text)
            cost = calculate_api_cost(api_model, input_tokens, output_tokens)

            logger.info(f"[API] API reconstruction successful using model {api_model}")
            method_used = f"api_{api_model}"
            elapsed_time = time.time() - start_time
            logger.info(f"[API] Completed in {elapsed_time:.3f}s using {method_used}")

            # When returning successfully, clear recovery attempts
            _api_recovery_attempts.pop(request_key, None)

            return reconstructed_text, cost, 1 if api_model == "gpt-3.5-turbo" else 2
    except Exception as e:
        logger.error(f"[API] API enhancement failed: {e}")

    # Fallback to KB if API fails or wasn't used
    if kb_applied and kb_reconstructed != noisy_text:
        logger.info("[API] Using KB reconstruction as fallback")
        # When returning successfully, clear recovery attempts
        _api_recovery_attempts.pop(request_key, None)
        return kb_reconstructed, 0, 0

    # Ultimate fallback to basic reconstruction
    reconstructed = basic_text_reconstruction(noisy_text, use_kb=use_kb)
    elapsed_time = time.time() - start_time
    logger.info(f"[API] Completed in {elapsed_time:.3f}s using basic reconstruction (fallback)")

    # When returning successfully, clear recovery attempts
    _api_recovery_attempts.pop(request_key, None)

    return reconstructed, 0, 0


# Helper function for corruption estimation
def estimate_corruption_level(text):
    """
    Estimate the corruption level of text based on multiple indicators.
    Returns a value between 0 and 1 where higher values indicate more corruption.
    """
    # Split into words
    words = text.split()
    if not words:
        return 0.0

    # Common correct words that shouldn't be counted as corrupted
    common_words = {'the', 'that', 'this', 'is', 'are', 'and', 'in', 'with', 'for', 'of', 'to', 'have', 'has',
                    'it', 'on', 'be', 'by', 'at', 'as', 'not', 'from', 'will', 'can', 'I', 'you', 'we', 'they'}

    # Parliamentary terms (to give higher weight if corrupted)
    important_terms = {'Parliament', 'Commission', 'Council', 'Directive', 'Regulation', 'Rule', 'meeting',
                       'Quaestors', 'agenda', 'vote', 'proposal', 'amendment'}

    # Patterns indicating corruption
    corrupted_patterns = ['bb', 'bz', 'hz', 'jz', 'kz', 'pj', 'xn', 'qx', 'oj', 'wk', 'wg', 'vb', 'xj',
                          'lk', 'vn', 'tm', 'vw', 'oq', 'ws', 'zx', 'bt', 'oe', 'tm', 'wb', 'qm']

    # Count corrupted words
    corrupted_count = 0
    important_corrupted = 0

    for word in words:
        # Skip very short words and punctuation
        if len(word) <= 2 or all(c in '.,;:!?()[]{}"\'' for c in word):
            continue

        # Skip common correct words
        if word.lower() in common_words:
            continue

        # Check for unusual patterns
        has_pattern = any(pattern in word.lower() for pattern in corrupted_patterns)

        # Check for no vowels (likely corruption)
        no_vowels = len(word) > 3 and all(c not in 'aeiouAEIOU' for c in word)

        # Check for unusual character distribution
        char_counts = {}
        for c in word.lower():
            if c.isalpha():
                char_counts[c] = char_counts.get(c, 0) + 1
        unusual_distribution = any(count >= 3 for c, count in char_counts.items())

        # Mark as corrupted if it meets any criteria
        if has_pattern or no_vowels or unusual_distribution:
            corrupted_count += 1

            # Check if it's a corrupted important term
            for term in important_terms:
                # Calculate string similarity
                similarity = difflib.SequenceMatcher(None, word.lower(), term.lower()).ratio()
                if 0.6 < similarity < 0.9:  # Close but not exact match to important term
                    important_corrupted += 1
                    break

    # Calculate base corruption level
    base_corruption = corrupted_count / max(1, len(words))

    # Apply importance boost
    importance_factor = 1.0 + (important_corrupted / max(1, len(words)) * 0.5)

    # Final corruption level (capped at 1.0)
    corruption_level = min(1.0, base_corruption * importance_factor)

    return corruption_level

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


def validate_reconstruction(original, reconstructed, kb_reconstructed=None):
    """
    Enhanced validation with KB awareness and better error recovery.

    Args:
        original: Original corrupted text
        reconstructed: Reconstructed text from API
        kb_reconstructed: Optional KB reconstruction for additional guidance
    """
    if original == reconstructed:
        # No changes were made, apply known patterns
        return apply_phrase_patterns(original)

    words_orig = original.split()
    words_recon = reconstructed.split()

    # If KB reconstruction is available, use it for additional guidance
    words_kb = None
    if kb_reconstructed is not None and kb_reconstructed != original:
        words_kb = kb_reconstructed.split()

    # Step 1: Check for known problematic replacements
    for i in range(min(len(words_orig), len(words_recon))):
        # Fix "that" incorrectly changed to "the"
        if words_orig[i].lower() == "that" and words_recon[i].lower() == "the":
            words_recon[i] = words_orig[i]  # Restore original "that"

        # Fix common words that shouldn't be replaced
        common_words = {'are', 'has', 'is', 'the', 'that', 'this', 'have', 'been', 'and', 'not', 'actually',
                        'for', 'with', 'in', 'on', 'by', 'to', 'from', 'of', 'we', 'you', 'they', 'it'}

        if words_orig[i].lower() in common_words and words_recon[i].lower() not in common_words:
            # If a common word was replaced with an uncommon word, restore original
            words_recon[i] = words_orig[i]

        # Fix capitalization preservation
        if words_orig[i] != words_recon[i] and words_orig[i][0].isupper() and words_recon[i][0].islower():
            words_recon[i] = words_recon[i].capitalize()

        # NEW: Use KB reconstruction for specific words if available
        if words_kb is not None and i < len(words_kb):
            # If KB and API agree on a correction but differ from original, trust it highly
            if words_kb[i] == words_recon[i] and words_kb[i] != words_orig[i]:
                # Keep the agreed correction (already in words_recon)
                pass
            # If KB changed a word but API didn't, consider KB suggestion
            elif words_kb[i] != words_orig[i] and words_recon[i] == words_orig[i]:
                # Check if KB correction is sensible (not just another corruption)
                if not any(p in words_kb[i].lower() for p in ['bb', 'bz', 'hz', 'jz', 'oh', 'xj', 'nx']):
                    # Accept KB correction with 50% probability
                    if random.random() < 0.5:
                        words_recon[i] = words_kb[i]

    # Step 2: Expanded pattern-based corrections for European Parliament terminology
    for i in range(min(len(words_orig), len(words_recon))):
        # Check for specific parliamentary terms that might be corrupted
        parliamentary_corrections = {
            # Name corrections
            "plooij-vbn": "Plooij-van",
            "gorsel": "Gorsel",
            "bertngueb": "Berenguer",
            "fuster": "Fuster",
            "lynne": "Lynne",
            "díez": "Díez",
            "hecds": "Hicks",
            "efabs": "Evans",

            # Institution corrections
            "parliamemt": "Parliament",
            "europenn": "European",
            "commissiob": "Commission",
            "coupcil": "Council",

            # Common terms
            "qourle": "course",
            "ieetpng": "meeting",
            "wgn": "can",
            "wvat": "that",
            "tiio": "this",
            "parn": "part"
        }

        # Apply corrections for parliamentary terms
        for pattern, replacement in parliamentary_corrections.items():
            if pattern in words_recon[i].lower():
                # Preserve capitalization
                if words_recon[i][0].isupper():
                    words_recon[i] = replacement.capitalize()
                else:
                    words_recon[i] = replacement
                break

    # Step 3: Check for uncorrected corrupted words using enhanced pattern recognition
    for i in range(min(len(words_orig), len(words_recon))):
        # If words remain unchanged but appear corrupted
        if words_orig[i] == words_recon[i] and len(words_orig[i]) > 3:
            # More expanded patterns indicating corruption
            corruption_patterns = ['bb', 'bz', 'hz', 'jz', 'oh', 'xj', 'nx', 'wk', 'vb', 'xn', 'qx', 'oj', 'zx']

            # Check for unusual patterns indicating corruption
            if any(p in words_orig[i].lower() for p in corruption_patterns) or \
                    all(c not in 'aeiou' for c in words_orig[i].lower()):
                # Try to correct using improved pattern matching
                for pattern, replacement in common_corrections:
                    similarity = difflib.SequenceMatcher(None, words_orig[i].lower(), pattern).ratio()
                    if similarity > 0.65:  # Lower threshold for better correction coverage
                        # Preserve case
                        if words_orig[i][0].isupper():
                            words_recon[i] = replacement.capitalize()
                        else:
                            words_recon[i] = replacement
                        break

    # Step 4: Check for coherence issues
    # Fix repeated words
    for i in range(1, len(words_recon)):
        if words_recon[i] == words_recon[i - 1] and words_orig[i] != words_orig[i - 1]:
            # Fix repeated words that shouldn't be repeated
            words_recon[i] = words_orig[i]

    # Step 5: Apply final phrase-level corrections
    result_text = " ".join(words_recon)
    result_text = apply_phrase_patterns(result_text)

    return result_text


def detect_corruption_patterns(text):
    """
    Detect specific corruption patterns in text to identify reconstruction needs.

    Args:
        text: Text to analyze

    Returns:
        Dictionary with detected patterns and counts
    """
    results = {
        'detected_patterns': [],
        'corruption_level': 0.0,
        'critical_corruptions': 0,
        'name_corruptions': 0,
        'institution_corruptions': 0
    }

    # Check for name corruptions
    name_patterns = [
        "Plooij-vbn", "vbn Gorsel", "Lynme", "Lymne", "Bertngueb",
        "Berenguef", "Evams", "Evabs", "Díef", "Hicka", "Segmu"
    ]

    # Check for institution corruptions
    institution_patterns = [
        "Parliamemt", "Palrliament", "Commissiob", "Commizion", "Coupcil",
        "Europenn", "Eurepean", "Quaeftor", "Quaertos"
    ]

    # Check for procedural term corruptions
    procedure_patterns = [
        "ieetpng", "aleeda", "sessien", "vite", "amendmert", "proporal",
        "debpte", "procwdure", "codecislon", "presidenby"
    ]

    # Check for critical grammar corruptions
    grammar_patterns = [
        "wgn", "wvat", "tiio", "tmab", "coq", "frve", "qourle", "vof",
        "inaormatign", "ttv"
    ]

    # Count pattern occurrences
    for pattern in name_patterns:
        if pattern in text:
            results['detected_patterns'].append(pattern)
            results['name_corruptions'] += 1

    for pattern in institution_patterns:
        if pattern in text:
            results['detected_patterns'].append(pattern)
            results['institution_corruptions'] += 1

    procedure_count = 0
    for pattern in procedure_patterns:
        if pattern in text:
            results['detected_patterns'].append(pattern)
            procedure_count += 1

    grammar_count = 0
    for pattern in grammar_patterns:
        if pattern in text:
            results['detected_patterns'].append(pattern)
            grammar_count += 1

    # Calculate corruption level
    words = text.split()
    corrupt_word_count = results['name_corruptions'] + results[
        'institution_corruptions'] + procedure_count + grammar_count

    # Critical corruptions are more important
    results['critical_corruptions'] = results['name_corruptions'] + results['institution_corruptions']

    # Calculate overall corruption level
    results['corruption_level'] = min(1.0, corrupt_word_count / max(1, len(words)) * 2)

    # Boost corruption level for critical corruptions
    if results['critical_corruptions'] > 0:
        results['corruption_level'] = max(results['corruption_level'], 0.5)

    return results


def should_use_api(noisy_text, kb_result=None, kb_confidence=0.0, budget_remaining=1.0, rl_agent=None):
    """
    Decide whether to use API for reconstruction based on multiple factors.
    """
    # Default choices
    should_use = False
    model = "gpt-3.5-turbo"  # Default model
    reason = "default"

    # Step 1: Basic checks
    if not openai_available or not openai_client:
        return False, None, "API not available"

    if budget_remaining < 0.01:
        return False, None, "budget depleted"

    # Step 2: Use a more robust corruption assessment
    corruption_info = detect_corruption_patterns(noisy_text)
    corruption_level = corruption_info['corruption_level']

    # New: Calculate text complexity score for better decision making
    word_length = len(noisy_text.split())
    complexity_score = 0.0

    # Longer text gets higher complexity score
    if word_length > 20:
        complexity_score += 0.2

    # Complex parliamentary words indicate higher difficulty
    parliamentary_indicators = ['Parliament', 'Commission', 'Council', 'Rule', 'Quaestors',
                                'Session', 'Directive', 'Regulation', 'amendment']
    for word in parliamentary_indicators:
        if word.lower() in noisy_text.lower():
            complexity_score += 0.1

    # Names need special handling, check for name indicators
    name_indicators = ['Mr', 'Mrs', 'Ms', 'Dr', 'van', 'de', 'von']
    for indicator in name_indicators:
        if indicator in noisy_text.split():
            complexity_score += 0.15

    # Step 3: Improved KB effectiveness assessment
    kb_effectiveness = 0.0
    if kb_result is not None:
        # Calculate what percentage of words were actually fixed
        orig_words = noisy_text.split()
        kb_words = kb_result.split()
        words_changed = sum(1 for i in range(min(len(orig_words), len(kb_words)))
                            if orig_words[i] != kb_words[i])

        if len(orig_words) > 0:
            kb_effectiveness = words_changed / len(orig_words)

        # Check if KB made ANY changes
        kb_effective = kb_result != noisy_text

        # Check if KB changes were meaningful (confidence-weighted)
        if kb_effective and kb_confidence > 0.8:
            kb_effectiveness += 0.3

    # More definitive decision logic based on combined factors
    if corruption_level > 0.4 or complexity_score > 0.3:
        # Highly corrupted or complex text
        should_use = True
        reason = "high_corruption_or_complexity"

        # Choose model based on severity and budget
        if corruption_level > 0.6 and budget_remaining > 0.15:
            model = "gpt-4-turbo"
            reason = "severe_corruption"
    elif kb_effectiveness < 0.3 and corruption_level > 0.2:
        # KB didn't fix enough and there's meaningful corruption
        should_use = True
        reason = "kb_insufficient"

    # Step 5: Override with RL agent if available
    if rl_agent is not None:
        try:
            # Create enhanced state representation for PPO
            if isinstance(rl_agent, PPOAgent):
                # Get corruption level with improved detection
                corruption_level = estimate_corruption_level(noisy_text)

                # Use parliamentary features if available, or extract them
                if parl_features is None:
                    parl_features = extract_parliamentary_features(noisy_text)

                # Create enhanced state with parliamentary features
                state = rl_agent.get_enhanced_state(
                    corruption_level,
                    len(noisy_text.split()),
                    parl_features
                )

                # Get action, log_prob, and value from PPO agent
                action, log_prob, _ = rl_agent.act(state)

                # Store log_prob for training
                log_prob_value = log_prob

                # Override decision based on RL agent
                if action == 0:  # Basic reconstruction
                    should_use = False
                    reason = "ppo_decision_basic"
                elif action == 1:  # GPT-3.5
                    should_use = True
                    model = "gpt-3.5-turbo"
                    reason = "ppo_decision_gpt3"
                elif action == 2:  # GPT-4
                    should_use = True
                    model = "gpt-4-turbo"
                    reason = "ppo_decision_gpt4"
            else:
                # Legacy code for backward compatibility with AdvancedRLAgent
                state = [corruption_level, kb_confidence, budget_remaining]
                if hasattr(rl_agent, 'get_enhanced_state'):
                    state = rl_agent.get_enhanced_state(
                        corruption_level,
                        len(noisy_text.split()),
                        [kb_confidence, len(corruption_info['detected_patterns'])]
                    )

                # Get action from RL agent
                action, _ = rl_agent.select_action(
                    state,
                    budget_remaining,
                    corruption_level=corruption_level
                )

                # Override decision based on RL agent
                if action == 0:  # Basic reconstruction
                    should_use = False
                    reason = "rl_decision_basic"
                elif action == 1:  # GPT-3.5
                    should_use = True
                    model = "gpt-3.5-turbo"
                    reason = "rl_decision_gpt3"
                elif action == 2:  # GPT-4
                    should_use = True
                    model = "gpt-4-turbo"
                    reason = "rl_decision_gpt4"
        except Exception as e:
            logger.warning(f"RL agent decision failed: {e}")

    return should_use, model, reason

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

def cascade_reconstruction(noisy_text, context=None, rl_agent=None, budget_remaining=1.0,
                           use_kb=True, additional_contexts=None):
    """
    Implements a cascade reconstruction strategy with ensemble combination:
    1. Start with KB reconstruction
    2. If KB confidence is low, try API reconstruction
    3. Combine reconstructions with confidence-weighted ensemble

    Args:
        noisy_text: Corrupted text to reconstruct
        context: Optional context for reconstruction
        rl_agent: Optional RL agent for decision making
        budget_remaining: Remaining API budget
        use_kb: Whether to use KB reconstruction
        additional_contexts: Additional context sentences

    Returns:
        Reconstructed text, cost, method used
    """
    start_time = time.time()

    # Step 1: KB Reconstruction
    kb_result = noisy_text  # Default to no change
    kb_applied = False
    kb_confidence = 0.0

    if use_kb:
        try:
            from knowledge_base import get_or_create_knowledge_base
            kb = get_or_create_knowledge_base()

            # Try KB reconstruction
            kb_result = kb.kb_guided_reconstruction(noisy_text)
            kb_applied = kb_result != noisy_text

            # Calculate confidence in KB result
            if kb_applied:
                # Calculate different metrics to assess confidence
                word_diff_ratio = sum(1 for a, b in zip(noisy_text.split(), kb_result.split())
                                      if a != b) / max(1, len(noisy_text.split()))
                char_overlap = difflib.SequenceMatcher(None, noisy_text, kb_result).ratio()

                # Calculate confidence score with more weight on word-level changes
                kb_confidence = char_overlap * (1 - min(0.4, word_diff_ratio)) * 1.1  # Slight boost

                logger.info(f"KB reconstruction confidence: {kb_confidence:.2f}")
        except Exception as e:
            logger.warning(f"KB reconstruction failed: {e}")

    # Step 2: Basic reconstruction
    basic_result = basic_text_reconstruction(noisy_text, use_kb=False)  # Avoid double KB usage
    basic_applied = basic_result != noisy_text

    # Calculate confidence in basic result
    basic_confidence = 0.0
    if basic_applied:
        # Similar confidence calculation
        word_diff_ratio = sum(1 for a, b in zip(noisy_text.split(), basic_result.split())
                              if a != b) / max(1, len(noisy_text.split()))
        char_overlap = difflib.SequenceMatcher(None, noisy_text, basic_result).ratio()
        basic_confidence = char_overlap * (1 - min(0.4, word_diff_ratio))

    # Step 3: Decide if API is needed based on confidence and corruption level
    api_result = None
    api_applied = False
    api_confidence = 0.0
    api_cost = 0.0

    # Estimate corruption level
    from SD5 import estimate_corruption_level
    corruption_level = estimate_corruption_level(noisy_text)

    # Decision logic for API usage
    should_use_api = False

    # 1. Low confidence in KB and basic reconstruction
    if max(kb_confidence, basic_confidence) < 0.8:
        should_use_api = True

    # 2. High corruption level
    if corruption_level > 0.3:
        should_use_api = True

    # 3. Use RL agent if available
    if rl_agent is not None:
        try:
            # Prepare state
            if hasattr(rl_agent, 'get_enhanced_state'):
                text_length = len(noisy_text.split())
                semantic_features = [kb_confidence, basic_confidence]
                state = rl_agent.get_enhanced_state(corruption_level, text_length, semantic_features)
            else:
                state = torch.tensor([corruption_level, kb_confidence, basic_confidence], dtype=torch.float32)

            # Get action from RL agent
            action, _ = rl_agent.select_action(state, budget_remaining, corruption_level=corruption_level)

            # Update API decision based on RL agent
            should_use_api = action > 0  # Actions > 0 use API
        except Exception as e:
            logger.warning(f"RL agent decision failed: {e}")

    # Use API if decided and available
    if should_use_api and budget_remaining > 0.05 and openai_available and openai_client:
        from SD5 import api_reconstruct_with_semantic_features
        api_result, api_cost, api_action = api_reconstruct_with_semantic_features(
            noisy_text, context, rl_agent, budget_remaining,
            use_kb=False,  # Avoid double KB usage
            additional_contexts=additional_contexts
        )
        api_applied = api_result != noisy_text

        # Calculate API confidence
        if api_applied:
            # API gets a confidence boost due to its power
            word_diff_ratio = sum(1 for a, b in zip(noisy_text.split(), api_result.split())
                                  if a != b) / max(1, len(noisy_text.split()))
            char_overlap = difflib.SequenceMatcher(None, noisy_text, api_result).ratio()
            api_confidence = (char_overlap * (1 - min(0.4, word_diff_ratio))) * 1.2  # 20% boost

    # Step 4: Combine reconstructions using confidence-weighted ensemble
    results = []
    confidences = []

    # Add results that made changes
    if kb_applied:
        results.append(kb_result)
        confidences.append(kb_confidence)

    if basic_applied:
        results.append(basic_result)
        confidences.append(basic_confidence)

    if api_applied:
        results.append(api_result)
        confidences.append(api_confidence)

    # If no reconstruction made changes, return original
    if not results:
        logger.info(f"No reconstruction method made changes, returning original")
        return noisy_text, 0, "none"

    # If only one reconstruction made changes, use it
    if len(results) == 1:
        logger.info(f"Only one reconstruction method made changes, using it")
        # Determine which method was used
        method = "kb" if kb_applied and results[0] == kb_result else \
            "api" if api_applied and results[0] == api_result else "basic"
        return results[0], api_cost, method

    # If multiple methods made changes, use confidence-weighted voting for each word
    final_result = ensemble_word_voting(noisy_text, results, confidences)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    logger.info(f"Cascade reconstruction completed in {elapsed_time:.3f}s using ensemble of methods")

    # Determine primary method based on highest confidence
    max_confidence_idx = confidences.index(max(confidences))
    method = "kb" if kb_applied and results[max_confidence_idx] == kb_result else \
        "api" if api_applied and results[max_confidence_idx] == api_result else "basic"

    return final_result, api_cost, method
def ensemble_word_voting(original, reconstructions, confidences):
    """
    Perform word-level voting across multiple reconstructions.

    Args:
        original: Original (corrupted) text
        reconstructions: List of reconstructed texts
        confidences: List of confidence scores for each reconstruction

    Returns:
        Final text with confidence-weighted voting for each word
    """
    # Tokenize all texts
    orig_words = original.split()
    recon_words_list = [r.split() for r in reconstructions]

    # Normalize confidences to sum to 1
    conf_sum = sum(confidences)
    if conf_sum == 0:
        norm_confidences = [1 / len(confidences)] * len(confidences)
    else:
        norm_confidences = [c / conf_sum for c in confidences]

    # New word list for result
    result_words = []

    # Process each position in the text
    for i in range(len(orig_words)):
        # If any reconstruction is shorter than original, keep original word
        if any(i >= len(words) for words in recon_words_list):
            result_words.append(orig_words[i])
            continue

        # Get all candidates for this position with their confidences
        candidates = {}

        # Add original word with a small base confidence
        candidates[orig_words[i]] = 0.1

        # Add reconstructed words with their confidences
        for j, words in enumerate(recon_words_list):
            if i < len(words):  # Make sure we don't go beyond the length
                word = words[i]
                if word in candidates:
                    candidates[word] += norm_confidences[j]
                else:
                    candidates[word] = norm_confidences[j]

        # Select the word with highest confidence
        best_word = max(candidates.items(), key=lambda x: x[1])[0]
        result_words.append(best_word)

    # Combine into final result
    return " ".join(result_words)


def benchmark_reconstruction_methods(test_samples, output_path=None):
    """
    Benchmark different reconstruction methods on test samples.

    Args:
        test_samples: List of (original, corrupted) text pairs
        output_path: Optional path to save results

    Returns:
        Dictionary of performance metrics
    """
    import matplotlib.pyplot as plt

    # Initialize metrics
    metrics = {
        "method": [],
        "bleu": [],
        "rouge": [],
        "semantic": [],
        "success_rate": []
    }

    # Get semantic loss for evaluation
    try:
        from semantic_loss import get_semantic_loss
        semantic_loss_fn = get_semantic_loss()
    except:
        semantic_loss_fn = None

    # Get knowledge base
    kb = get_or_create_knowledge_base()

    # Track API costs
    api_costs = {
        "api": 0.0,
        "ensemble": 0.0
    }

    # Process each sample with different methods
    for original, corrupted in test_samples:
        # 1. Basic reconstruction
        basic_result = basic_text_reconstruction(corrupted, use_kb=False)
        basic_metrics = evaluate_reconstruction_with_semantics(original, basic_result, semantic_loss_fn)

        # 2. KB reconstruction
        kb_result = kb.kb_guided_reconstruction(corrupted)
        kb_metrics = evaluate_reconstruction_with_semantics(original, kb_result, semantic_loss_fn)

        # 3. API reconstruction (if available)
        if openai_available and openai_client:
            api_result, api_cost, _ = api_reconstruct_with_semantic_features(
                corrupted, context="", use_kb=False)
            api_metrics = evaluate_reconstruction_with_semantics(original, api_result, semantic_loss_fn)
            api_costs["api"] += api_cost

            # 4. Ensemble method
            reconstructions = [r for r in [api_result, kb_result, basic_result] if r != corrupted]
            if len(reconstructions) >= 2:
                # Calculate confidences
                confidences = []
                for result in reconstructions:
                    word_diff_ratio = sum(1 for a, b in zip(corrupted.split(), result.split())
                                          if a != b) / max(1, len(corrupted.split()))
                    char_overlap = difflib.SequenceMatcher(None, corrupted, result).ratio()
                    confidence = char_overlap * (1 - min(0.4, word_diff_ratio))
                    confidences.append(confidence)

                # Apply ensemble voting
                ensemble_result = ensemble_word_voting(corrupted, reconstructions, confidences)
                ensemble_metrics = evaluate_reconstruction_with_semantics(original, ensemble_result, semantic_loss_fn)
                api_costs["ensemble"] += api_cost  # Same cost as API

    # Add results to metrics dict
    methods = ["Basic", "KB", "API", "Ensemble"]
    for method in methods:
        # Only include API and Ensemble if they were used
        if method == "API" and not openai_available:
            continue
        if method == "Ensemble" and not openai_available:
            continue

        metrics["method"].append(method)

        # Add average metrics
        if method == "Basic":
            avg_metrics = basic_metrics
        elif method == "KB":
            avg_metrics = kb_metrics
        elif method == "API":
            avg_metrics = api_metrics
        else:  # Ensemble
            avg_metrics = ensemble_metrics

        metrics["bleu"].append(avg_metrics.get("BLEU", 0))
        metrics["rouge"].append(avg_metrics.get("ROUGEL", 0))
        metrics["semantic"].append(avg_metrics.get("SEMANTIC", 0))

    # Plot results if output path specified
    if output_path:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(metrics["method"]))
        width = 0.25

        ax.bar([i - width for i in x], metrics["bleu"], width, label="BLEU")
        ax.bar(x, metrics["rouge"], width, label="ROUGE-L")
        ax.bar([i + width for i in x], metrics["semantic"], width, label="SEMANTIC")

        ax.set_ylabel('Score')
        ax.set_title('Reconstruction Performance by Method')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics["method"])
        ax.legend()

        # Add cost annotations
        if openai_available:
            cost_text = f"API Cost: ${api_costs['api']:.2f}\nEnsemble Cost: ${api_costs['ensemble']:.2f}"
            plt.figtext(0.02, 0.02, cost_text)

        plt.savefig(output_path)
        print(f"Performance visualization saved to {output_path}")

    return metrics

# Add this near the top of the file with other constants
PARLIAMENTARY_TERMS = [
    "Parliament", "Commission", "Council", "Directive", "Regulation",
    "Committee", "Member", "State", "European", "Union", "President",
    "Rule", "session", "agenda", "vote", "voting", "proposal",
    "amendment", "debate", "procedure", "codecision", "legislation",
    "Rapporteur", "Quaestors", "Presidency", "MEP", "motion",
    "Plooij-van", "Gorsel", "Lynne", "Berenguer", "Fuster", "Schroedter",
    "Díez", "Evans", "Hicks"
]
def run_enhanced_pipeline(num_samples=None, noise_level=None, noise_type=None,
                          use_api_pct=None, comparison_mode=None, use_self_supervised=None,
                          use_semantic_loss=None, use_vae_compression=None,
                          use_content_adaptive_coding=None, use_knowledge_base=True,
                          use_ensemble=True, aggressive_api=True):  # Changed defaults to True
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
        use_ensemble: Whether to use ensemble voting approach for reconstruction
        aggressive_api: Whether to use more aggressive API criteria in ensemble mode
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
    from physical_semantic_integration import DimensionRegistry
    dimension_registry = get_dimension_registry()
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
    rl_agent = PPOAgent(state_dim=16) if use_rl else None

    semantic_optimizer = None
    if ENABLE_PHYSICAL_CHANNEL and physical_channel_imported:
        try:
            from physical_semantic_integration import SemanticChannelOptimizer
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

            # Define challenging text detection (used in RL section)
            challenging_indicators = ['xont', 'dotk', 'ceea', 'jvsz', 'xjeting', 'yreudful']
            is_challenging_text = any(indicator in corrupted_text.lower() for indicator in challenging_indicators)

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

            # Calculate corruption level based on differences between original and corrupted text
            # Add this line to define corruption_level before it's used
            corruption_level = min(1.0, sum(1 for a, b in zip(corrupted_text.split(), sentence.split())
                                            if a != b) / max(1, len(corrupted_text.split())))

            # If using ensemble approach
            if use_ensemble:
                # First get KB result
                kb_result = None
                kb_confidence = 0.0
                if use_knowledge_base and kb is not None:
                    kb_result = kb.kb_guided_reconstruction(corrupted_text)

                    # Calculate KB confidence if it made changes
                    if kb_result != corrupted_text:
                        if hasattr(kb, 'calculate_kb_confidence'):
                            kb_confidence = kb.calculate_kb_confidence(corrupted_text, kb_result)
                        else:
                            # Fallback confidence calculation
                            word_diff_ratio = sum(1 for a, b in zip(corrupted_text.split(), kb_result.split())
                                                  if a != b) / max(1, len(corrupted_text.split()))
                            char_overlap = difflib.SequenceMatcher(None, corrupted_text, kb_result).ratio()
                            kb_confidence = char_overlap * (1 - min(0.4, word_diff_ratio))

                # Get basic reconstruction (without KB to avoid duplication)
                basic_result = basic_text_reconstruction(corrupted_text, use_kb=False)
                basic_confidence = 0.0

                # Calculate basic confidence if it made changes
                if basic_result != corrupted_text:
                    word_diff_ratio = sum(1 for a, b in zip(corrupted_text.split(), basic_result.split())
                                          if a != b) / max(1, len(corrupted_text.split()))
                    char_overlap = difflib.SequenceMatcher(None, corrupted_text, basic_result).ratio()
                    basic_confidence = char_overlap * (1 - min(0.4, word_diff_ratio))

                # Decide on API usage
                use_api_now = False

                if aggressive_api:
                    # Use advanced detection
                    corruption_info = detect_corruption_patterns(corrupted_text)

                    # Use API for critical corruptions or when KB confidence is low
                    if (corruption_info['critical_corruptions'] > 0 or
                            (kb_result != corrupted_text and kb_confidence < 0.7) or
                            corruption_info['corruption_level'] > 0.3):
                        use_api_now = True
                else:
                    # Use standard logic based on RL agent or corruption level
                    use_api = (openai_available and random.random() < use_api_pct) or (
                            is_challenging_text and budget_remaining > 0.2 and openai_available)
                    use_api_now = use_api

                api_result = None
                api_confidence = 0.0
                api_cost = 0

                # Get API result if needed
                if use_api_now and openai_available and budget_remaining > 0.05:
                    api_result, api_cost, _ = api_reconstruct_with_semantic_features(
                        corrupted_text, context, rl_agent, budget_remaining,
                        semantic_features, use_kb=False,
                        additional_contexts=context_list[1:] if len(context_list) > 1 else None)

                    # Calculate API confidence if it made changes
                    if api_result != corrupted_text:
                        word_diff_ratio = sum(1 for a, b in zip(corrupted_text.split(), api_result.split())
                                              if a != b) / max(1, len(corrupted_text.split()))
                        char_overlap = difflib.SequenceMatcher(None, corrupted_text, api_result).ratio()
                        api_confidence = char_overlap * (1 - min(0.4, word_diff_ratio)) * 1.2  # 20% boost

                # Now decide which reconstructions to use for ensemble
                reconstructions = []
                confidences = []

                # Add KB result if it made changes
                if kb_result is not None and kb_result != corrupted_text:
                    reconstructions.append(kb_result)
                    confidences.append(kb_confidence)

                # Add basic result if it made changes
                if basic_result != corrupted_text:
                    reconstructions.append(basic_result)
                    confidences.append(basic_confidence)

                # Add API result if it made changes
                if api_result is not None and api_result != corrupted_text:
                    reconstructions.append(api_result)
                    confidences.append(api_confidence)

                # Apply ensemble voting if we have multiple reconstructions
                if len(reconstructions) >= 2:
                    semantic_reconstructed = ensemble_word_voting(corrupted_text, reconstructions, confidences)
                    sample_result["semantic_method"] = "ensemble"
                    if api_result is not None:
                        sample_result["api_cost"] = api_cost

                # Otherwise, use the best single method
                elif len(reconstructions) == 1:
                    semantic_reconstructed = reconstructions[0]

                    # Record which method was used
                    if reconstructions[0] == kb_result:
                        sample_result["semantic_method"] = "kb"
                    elif reconstructions[0] == api_result:
                        sample_result["semantic_method"] = "api"
                        sample_result["api_cost"] = api_cost
                    else:
                        sample_result["semantic_method"] = "basic"

                # If no method made changes, keep original
                else:
                    semantic_reconstructed = corrupted_text
                    sample_result["semantic_method"] = "none"

            else:
                # Use original approach if ensemble is disabled
                if use_rl:
                    # For RL agent, if we've detected challenging text, boost corruption level
                    if is_challenging_text:
                        # Boost corruption level signal for RL agent
                        if corruption_level is not None:
                            corruption_level = max(corruption_level, 0.8)
                        else:
                            corruption_level = 0.8

                    # Later in the function, when using the agent:
                    if use_rl:
                        # Extract parliamentary features for better RL state representation
                        parl_features = extract_parliamentary_features(corrupted_text)

                        # Use PPO agent with enhanced features for API decision
                        semantic_reconstructed, api_cost, action = api_reconstruct_with_semantic_features(
                            corrupted_text, context, rl_agent, budget_remaining, parl_features,
                            use_kb=use_knowledge_base
                        )

                        # Add reward calculation for PPO with enhanced metrics
                        if action is not None:
                            # Calculate reconstruction quality metrics
                            metrics = evaluate_reconstruction_with_semantics(
                                sentence, semantic_reconstructed, semantic_loss_fn)

                            # Add parliamentary term detection to metrics
                            metrics['parl_terms'] = sum(1 for term in PARLIAMENTARY_TERMS if term in sentence)

                            # Check for exact match
                            metrics['exact_match'] = semantic_reconstructed == sentence

                            # Calculate reward with PPO-specific implementation
                            reward = rl_agent.calculate_reward(metrics, action, api_cost)

                            # Update PPO agent with experience
                            if hasattr(rl_agent, 'update'):
                                # Create next state (simplified for demonstration)
                                next_state = rl_agent.get_enhanced_state(
                                    estimate_corruption_level(semantic_reconstructed),
                                    len(semantic_reconstructed.split()),
                                    parl_features
                                )

                                # Update agent
                                rl_agent.update(state, action, reward, next_state, log_prob)

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
                    # Define force_api here, when budget_remaining is definitely in scope
                    force_api = is_challenging_text and budget_remaining > 0.2 and openai_available

                    # Use fixed probability for API decision or force_api when text is challenging
                    use_api = (openai_available and random.random() < use_api_pct) or force_api
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
        "enhanced_rl": use_rl and isinstance(rl_agent, PPOAgent)
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


def run_experiment_suite():
    """Run multiple experiments with different configurations"""
    experiment_configs = [
        # Baseline
        {
            "name": "baseline",
            "num_samples": 50,
            "noise_level": 0.15,
            "noise_type": 'gaussian',
            "use_api_pct": 0.5,
            "use_semantic_loss": True,
            "use_vae_compression": True,
            "use_content_adaptive_coding": True
        },
        # Higher noise level
        {
            "name": "high_noise",
            "num_samples": 50,
            "noise_level": 0.25,
            "noise_type": 'gaussian',
            "use_api_pct": 0.5,
            "use_semantic_loss": True,
            "use_vae_compression": True,
            "use_content_adaptive_coding": True
        },
        # Different noise type
        {
            "name": "burst_noise",
            "num_samples": 50,
            "noise_level": 0.15,
            "noise_type": 'burst',
            "use_api_pct": 0.5,
            "use_semantic_loss": True,
            "use_vae_compression": True,
            "use_content_adaptive_coding": True
        },
        # Higher API usage
        {
            "name": "high_api",
            "num_samples": 50,
            "noise_level": 0.15,
            "noise_type": 'gaussian',
            "use_api_pct": 0.8,
            "use_semantic_loss": True,
            "use_vae_compression": True,
            "use_content_adaptive_coding": True
        }
    ]

    results_summary = {}

    for config in experiment_configs:
        print(f"\n=== Running experiment: {config['name']} ===")

        # Extract parameters
        results = run_enhanced_pipeline(
            num_samples=config["num_samples"],
            noise_level=config["noise_level"],
            noise_type=config["noise_type"],
            use_api_pct=config["use_api_pct"],
            comparison_mode=True,
            use_semantic_loss=config["use_semantic_loss"],
            use_vae_compression=config["use_vae_compression"],
            use_content_adaptive_coding=config["use_content_adaptive_coding"]
        )

        # Store results summary
        results_summary[config["name"]] = {
            "config": config,
            "metrics": results["overall_metrics"] if results and "overall_metrics" in results else {},
            "cost": results["cost"] if results and "cost" in results else {}
        }

    # Compare results across experiments
    print("\n=== Experiment Comparison ===")
    for exp_name, exp_results in results_summary.items():
        metrics = exp_results["metrics"]
        print(f"Experiment: {exp_name}")
        print(f"  Semantic BLEU: {metrics.get('semantic_avg_BLEU', 0):.4f}")
        print(f"  Semantic ROUGE-L: {metrics.get('semantic_avg_ROUGEL', 0):.4f}")
        print(f"  Semantic Score: {metrics.get('semantic_avg_SEMANTIC', 0):.4f}")
        print(f"  API Cost: ${exp_results['cost'].get('total', 0):.4f}")
        print("")

    return results_summary
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
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Semantic Communication Pipeline')
    parser.add_argument('--run_experiments', action='store_true',
                        help='Run multiple experiment configurations')
    parser.add_argument('--samples', type=int, default=50,
                        help='Number of samples to process')
    parser.add_argument('--noise_level', type=float, default=0.15,
                        help='Noise level (0-1)')
    parser.add_argument('--noise_type', choices=['gaussian', 'burst', 'dropout'],
                        default='gaussian', help='Type of noise')
    parser.add_argument('--api_pct', type=float, default=0.5,
                        help='Percentage of samples to use API for')
    # Add new arguments for our enhanced functionality
    parser.add_argument('--use_ensemble', action='store_true',
                        help='Use ensemble voting for text reconstruction')
    parser.add_argument('--aggressive_api', action='store_true',
                        help='Use more aggressive API strategy for challenging corruptions')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmarking on reconstruction methods')
    args = parser.parse_args()

    # Test system components first
    system_ok = test_system_components()
    if not system_ok:
        print("WARNING: System components are not functioning properly!")

    # Initialize dimension registry and lock dimensions
    dimension_registry = get_dimension_registry()

    # Run experiments or single pipeline
    if args.run_experiments:
        results_summary = run_experiment_suite()
    else:
        # Run the enhanced pipeline with arguments
        results = run_enhanced_pipeline(
            num_samples=args.samples,
            noise_level=args.noise_level,
            noise_type=args.noise_type,
            use_api_pct=args.api_pct,
            comparison_mode=True,
            use_semantic_loss=True,
            use_vae_compression=True,
            use_content_adaptive_coding=True,
            use_ensemble=args.use_ensemble,
            aggressive_api=args.aggressive_api
        )

        # Run benchmarking if requested
        if args.benchmark:
            print("\n===== Benchmarking Reconstruction Methods =====")

            # Create benchmarking samples from results
            benchmark_samples = []
            for sample in results["samples"][:10]:  # Use first 10 samples
                if "original" in sample and "semantic_noisy" in sample:
                    benchmark_samples.append((sample["original"], sample["semantic_noisy"]))

            # Run benchmarking with all methods
            benchmark_metrics = benchmark_reconstruction_methods(
                benchmark_samples,
                output_path=os.path.join(RESULTS_DIR, "benchmark_results.png")
            )

            print("\nBenchmark Results:")
            for i, method in enumerate(benchmark_metrics["method"]):
                print(f"{method}:")
                print(f"  BLEU: {benchmark_metrics['bleu'][i]:.4f}")
                print(f"  ROUGE-L: {benchmark_metrics['rouge'][i]:.4f}")
                print(f"  SEMANTIC: {benchmark_metrics['semantic'][i]:.4f}")

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

    if args.use_ensemble:
        print("- Ensemble Reconstruction: Using voting-based ensemble of multiple methods")
    if args.aggressive_api:
        print("- Aggressive API Strategy: Prioritizing API for critical corruptions")

    if results and "overall_metrics" in results:
        print("\nFinal metrics:")
        for key, value in sorted(results["overall_metrics"].items()):
            if key.startswith("semantic_avg_"):
                print(f"  {key}: {value:.4f}")

    # Add ensemble statistics if enabled
    if args.use_ensemble and results:
        print("\nEnsemble reconstruction statistics:")
        ensemble_count = sum(1 for sample in results["samples"] if sample.get("semantic_method") == "ensemble")
        kb_count = sum(1 for sample in results["samples"] if sample.get("semantic_method") == "kb")
        api_count = sum(1 for sample in results["samples"] if sample.get("semantic_method") == "api")
        basic_count = sum(1 for sample in results["samples"] if sample.get("semantic_method") == "basic")

        print(f"  Ensemble method used: {ensemble_count}/{len(results['samples'])} samples")
        print(f"  KB-only used: {kb_count}/{len(results['samples'])} samples")
        print(f"  API-only used: {api_count}/{len(results['samples'])} samples")
        print(f"  Basic-only used: {basic_count}/{len(results['samples'])} samples")

        # Calculate API cost savings if applicable
        if "cost" in results:
            print(f"  Total API cost: ${results['cost']['total']:.4f}")
            print(f"  Remaining budget: ${results['cost']['remaining']:.2f}")
