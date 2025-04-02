# knowledge_base.py
import os
import json
import pickle
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import difflib  # Built-in Python library for string comparisons

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextEmbeddingMapper:
    """Maps between text and embedding spaces using reference examples and nearest neighbors"""

    def __init__(self, reference_embeddings=None, reference_texts=None, k_neighbors=5):
        self.reference_embeddings = None
        self.reference_texts = None
        self.k_neighbors = k_neighbors
        self.nn_model = None

        # Initialize with references if provided
        if reference_embeddings is not None and reference_texts is not None:
            self.fit(reference_embeddings, reference_texts)

    def fit(self, embeddings, texts):
        """Fit the mapper with reference embeddings and texts"""
        if len(embeddings) != len(texts):
            raise ValueError(f"Mismatch between embeddings ({len(embeddings)}) and texts ({len(texts)})")

        # Store references
        self.reference_embeddings = np.array(embeddings)
        self.reference_texts = texts

        # Initialize nearest neighbor model
        self.nn_model = NearestNeighbors(n_neighbors=self.k_neighbors, algorithm='auto', metric='cosine')
        self.nn_model.fit(self.reference_embeddings)

        return self

    def embedding_to_text(self, embedding):
        """Convert embedding to approximate text representation"""
        if self.nn_model is None:
            raise ValueError("Mapper not fitted. Call fit() with reference data first.")

        # Find k nearest neighbors
        distances, indices = self.nn_model.kneighbors([embedding], n_neighbors=self.k_neighbors)

        # Weighted combination of nearest texts based on similarity
        weights = 1.0 - distances[0]  # Convert distance to similarity
        weights = weights / weights.sum()  # Normalize

        # Create weighted combination of texts
        # In a real system, this would be more sophisticated, potentially using
        # an actual language model to combine the texts coherently
        neighbors_text = [self.reference_texts[idx] for idx in indices[0]]

        # For now, just return the closest text
        closest_idx = indices[0][0]
        return self.reference_texts[closest_idx]

    def text_to_embedding(self, text, bert_model=None, bert_tokenizer=None):
        """Convert text back to embedding space"""
        if bert_model is not None and bert_tokenizer is not None:
            # Use BERT to get embedding if models provided
            inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(bert_model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = bert_model(**inputs)
                # Use mean of token embeddings as text embedding
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()

            return embedding

        # If BERT not available, use nearest neighbor approach
        if self.nn_model is None:
            raise ValueError("Mapper not fitted and no BERT model provided")

        # Find text match using string similarity
        from difflib import SequenceMatcher

        scores = [SequenceMatcher(None, text.lower(), ref.lower()).ratio()
                  for ref in self.reference_texts]
        best_idx = np.argmax(scores)

        return self.reference_embeddings[best_idx]


class KnowledgeEnhancedDecoder(nn.Module):
    """Neural decoder that enhances embeddings with knowledge base guidance"""

    def __init__(self, embedding_dim, hidden_dim=512, kb=None, mapper=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.kb = kb
        self.mapper = mapper

        # Neural network to enhance embeddings
        self.enhancement_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )

        # Attention mechanism for KB-guided enhancement
        self.query_proj = nn.Linear(embedding_dim, hidden_dim)
        self.key_proj = nn.Linear(embedding_dim, hidden_dim)
        self.value_proj = nn.Linear(embedding_dim, hidden_dim)
        self.attention_out = nn.Linear(hidden_dim, embedding_dim)

    def initialize_kb_embeddings(self, bert_model, bert_tokenizer):
        """Initialize embeddings for KB terms"""
        if self.kb is None:
            return

        # Get embeddings for all terms in KB
        term_embeddings = {}

        for term, correction in self.kb.term_dict.items():
            # Get embedding for original term
            inputs = bert_tokenizer(term, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(bert_model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = bert_model(**inputs)
                term_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()

            # Get embedding for corrected term
            inputs = bert_tokenizer(correction, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(bert_model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = bert_model(**inputs)
                corr_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()

            # Store both embeddings
            term_embeddings[term] = {
                'term_embedding': term_emb,
                'correction_embedding': corr_emb,
                'correction': correction
            }

        # Add to KB
        self.kb.term_embeddings = term_embeddings

    def get_kb_guided_embedding(self, embedding, text=None):
        """Get KB-guided enhancement for embedding"""
        if self.kb is None or not hasattr(self.kb, 'term_embeddings') or not self.kb.term_embeddings:
            return embedding

        # If text not provided but mapper is available, try to get text
        if text is None and self.mapper is not None:
            try:
                text = self.mapper.embedding_to_text(embedding)
            except Exception as e:
                logger.debug(f"Could not map embedding to text: {e}")
                return embedding

        if text is None:
            # Without text, fall back to embedding similarity
            return self._embedding_similarity_enhancement(embedding)
        else:
            # With text, do actual KB lookup and enhancement
            return self._text_guided_enhancement(embedding, text)

    def _embedding_similarity_enhancement(self, embedding):
        """Enhance embedding based on similarity to KB term embeddings"""
        # Find most similar terms in KB
        similarities = []
        correction_embeddings = []

        for term, data in self.kb.term_embeddings.items():
            term_emb = data['term_embedding']
            corr_emb = data['correction_embedding']

            # Calculate cosine similarity
            sim = np.dot(embedding, term_emb) / (
                    np.linalg.norm(embedding) * np.linalg.norm(term_emb))

            if sim > 0.7:  # Only consider high similarity terms
                similarities.append(sim)
                correction_embeddings.append(corr_emb)

        if not similarities:
            return embedding

        # Weight correction embeddings by similarity
        weights = np.array(similarities)
        weights = weights / weights.sum()

        # Weighted sum of correction embeddings
        correction_blend = np.zeros_like(embedding)
        for i, corr_emb in enumerate(correction_embeddings):
            correction_blend += weights[i] * corr_emb

        # Blend with original embedding
        blend_ratio = max(similarities)  # Higher similarity = more correction
        enhanced = (1 - blend_ratio) * embedding + blend_ratio * correction_blend

        # Normalize to preserve norm
        enhanced = enhanced / np.linalg.norm(enhanced) * np.linalg.norm(embedding)

        return enhanced

    def _text_guided_enhancement(self, embedding, text):
        """Enhance embedding using text and KB guidance"""
        # Apply KB text correction
        corrected_text = self.kb.kb_guided_reconstruction(text)

        # If no correction was made, return original
        if corrected_text == text:
            return embedding

        # Get embeddings for both original and corrected text
        # In a real implementation, you would use a text encoder here
        # For now, we'll use existing KB embeddings for approximation

        # Tokenize text
        tokens = text.lower().split()
        corrected_tokens = corrected_text.lower().split()

        # Find modifications
        modified_idxs = [i for i, (t1, t2) in enumerate(zip(tokens, corrected_tokens))
                         if t1 != t2 and t1 in self.kb.term_dict]

        if not modified_idxs:
            return embedding

        # Get correction vector - average of term correction vectors
        correction_vector = np.zeros_like(embedding)
        count = 0

        for idx in modified_idxs:
            if idx < len(tokens):
                term = tokens[idx]
                if term in self.kb.term_embeddings:
                    term_data = self.kb.term_embeddings[term]
                    term_emb = term_data['term_embedding']
                    corr_emb = term_data['correction_embedding']

                    # Calculate correction vector
                    term_correction = corr_emb - term_emb
                    correction_vector += term_correction
                    count += 1

        if count > 0:
            correction_vector /= count

            # Apply correction with attenuation
            # The more terms corrected, the stronger the correction
            correction_strength = min(0.5, count / len(tokens) * 0.8)
            enhanced = embedding + correction_strength * correction_vector

            # Normalize
            enhanced = enhanced / np.linalg.norm(enhanced) * np.linalg.norm(embedding)
            return enhanced

        return embedding

    def forward(self, embedding_batch, text_batch=None):
        """
        Forward pass with knowledge enhancement

        Args:
            embedding_batch: Tensor of embeddings [batch_size, embedding_dim]
            text_batch: Optional list of texts corresponding to embeddings

        Returns:
            Enhanced embeddings [batch_size, embedding_dim]
        """
        batch_size = embedding_batch.shape[0]
        device = embedding_batch.device

        # Convert to numpy for KB processing
        embeddings_np = embedding_batch.detach().cpu().numpy()

        # Process each embedding with KB guidance
        enhanced_embeddings = np.zeros_like(embeddings_np)

        for i in range(batch_size):
            if text_batch is not None and i < len(text_batch):
                text = text_batch[i]
            else:
                text = None

            # Get KB-guided enhancement
            enhanced_embeddings[i] = self.get_kb_guided_embedding(embeddings_np[i], text)

        # Convert back to torch tensor
        enhanced_tensor = torch.tensor(enhanced_embeddings, dtype=torch.float32).to(device)

        # Concatenate original and enhanced embeddings
        concat_embeddings = torch.cat([embedding_batch, enhanced_tensor], dim=1)

        # Apply neural enhancement
        output = self.enhancement_net(concat_embeddings)

        # Residual connection
        final_output = embedding_batch + output

        return final_output
class SemanticKnowledgeBase:
    """
    Knowledge base for semantic communication with synchronized encoder/decoder terminology.
    Supports term definitions, contextual relationships, and fuzzy matching.
    """

    def __init__(self, domain="europarl", kb_path=None, initialize=True):
        """Initialize knowledge base with optional domain-specific data"""
        self.domain = domain
        self.term_dict = {}  # Direct term mappings
        self.context_rules = {}  # Context-based corrections
        self.term_embeddings = {}  # Semantic embeddings of terms
        self.semantic_relations = {}  # Related terms

        # Set default path if not provided
        if kb_path is None:
            kb_path = os.path.join("./models", f"{domain}_knowledge_base.pkl")
        self.kb_path = kb_path

        # Try to load existing KB or initialize a new one
        if initialize:
            self.initialize()

    def initialize(self):
        """Initialize the knowledge base from file or with default data"""
        if os.path.exists(self.kb_path):
            try:
                with open(self.kb_path, "rb") as f:
                    kb_data = pickle.load(f)
                    self.term_dict = kb_data.get("term_dict", {})
                    self.context_rules = kb_data.get("context_rules", {})
                    self.term_embeddings = kb_data.get("term_embeddings", {})
                    self.semantic_relations = kb_data.get("semantic_relations", {})
                logger.info(f"Loaded knowledge base from {self.kb_path} with {len(self.term_dict)} terms")
                return True
            except Exception as e:
                logger.warning(f"Failed to load knowledge base: {e}")

        # Initialize with default data if loading failed
        if self.domain == "europarl":
            self._initialize_europarl_kb()
        else:
            # Create empty KB for other domains
            logger.info(f"Created new empty knowledge base for domain: {self.domain}")

        return self.save()

    def _initialize_europarl_kb(self):
        """Initialize with Europarl-specific knowledge"""
        logger.info("Initializing Europarl knowledge base")

        # Parliamentary terminology corrections
        self.term_dict = {
            # Basic errors
            "wkulz": "would",
            "couvsc": "course",
            "principdas": "principles",
            "accordancg": "accordance",
            "ymus": "your",
            "mnvice": "advice",
            "Rcne": "Rule",
            "acvioe": "advice",

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

        # Context rules for common parliamentary phrases
        self.context_rules = {
            "rule": {
                "context_words": ["procedure", "parliament", "143"],
                "candidates": ["Rule"]
            },
            "parliament": {
                "context_words": ["european", "member", "session"],
                "candidates": ["Parliament", "parliament", "Parliamentary", "parliamentary"]
            },
            "directive": {
                "context_words": ["european", "commission", "regulation"],
                "candidates": ["Directive", "directive", "regulation", "Regulation"]
            }
        }

        # Add semantic relations (related terms)
        self.semantic_relations = {
            "parliament": ["assembly", "chamber", "congress", "house"],
            "commission": ["committee", "delegation", "body", "panel"],
            "regulation": ["directive", "law", "legislation", "rule", "act"],
            "debate": ["discussion", "deliberation", "discourse", "exchange"],
            "vote": ["ballot", "poll", "referendum", "election"]
        }

        logger.info(
            f"Initialized Europarl KB with {len(self.term_dict)} terms, {len(self.context_rules)} context rules")

    def save(self):
        """Save knowledge base to file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.kb_path), exist_ok=True)

            # Save data
            kb_data = {
                "term_dict": self.term_dict,
                "context_rules": self.context_rules,
                "term_embeddings": self.term_embeddings,
                "semantic_relations": self.semantic_relations
            }

            with open(self.kb_path, "wb") as f:
                pickle.dump(kb_data, f)

            logger.info(f"Saved knowledge base to {self.kb_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")
            return False

    def add_term(self, term, correction, context=None, embedding=None, related_terms=None):
        """Add a term to the knowledge base"""
        self.term_dict[term] = correction

        if context:
            self.context_rules[term] = context

        if embedding is not None:
            self.term_embeddings[term] = embedding

        if related_terms:
            self.semantic_relations[term] = related_terms

        return True

    def query(self, text, max_results=5):
        """Query the knowledge base for relevant entries based on text"""
        tokens = text.lower().split()
        results = {}

        # Check direct term matches
        for token in tokens:
            if token in self.term_dict:
                results[token] = {
                    "correction": self.term_dict[token],
                    "confidence": 1.0,
                    "type": "direct"
                }

        # Check context rules
        for term, rule in self.context_rules.items():
            if term in tokens:
                # Count context words that appear in the text
                context_matches = sum(1 for word in rule["context_words"] if word in tokens)
                if context_matches > 0:
                    confidence = min(1.0, context_matches / len(rule["context_words"]))
                    results[term] = {
                        "candidates": rule["candidates"],
                        "confidence": confidence,
                        "type": "context"
                    }

        # Limit results
        sorted_results = {k: results[k] for k in sorted(results, key=lambda x: -results[x]["confidence"])[:max_results]}
        return sorted_results

    def _fuzzy_match(self, word, threshold=0.75):
        """Find closest matching term using difflib with optimizations"""
        import difflib

        # Common error patterns in parliamentary text
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

        # Try similarity to common errors
        for error, correction in common_errors.items():
            score = difflib.SequenceMatcher(None, word.lower(), error.lower()).ratio()
            if score > threshold:
                return correction, score

        # Try dictionary terms - first filter by first letter for efficiency
        best_match = None
        best_score = 0

        first_char = word[0].lower() if word else ''
        for term, correction in self.term_dict.items():
            if not term or term[0].lower() != first_char:
                continue

            score = difflib.SequenceMatcher(None, word.lower(), term.lower()).ratio()
            if score > threshold and score > best_score:
                best_match = correction
                best_score = score

        return best_match, best_score

    def _pattern_based_correction(self, word):
        """Apply pattern-based corrections for words not caught by other methods"""
        vowels = "aeiou"
        consonants = "bcdfghjklmnpqrstvwxyz"

        # Skip short words
        if len(word) <= 3:
            return None

        # Check for unusual patterns
        vowel_count = sum(1 for c in word.lower() if c in vowels)
        if vowel_count == 0 and len(word) > 3:
            # No vowels - try inserting common vowels
            for i in range(1, len(word) - 1):
                for v in "aeiou":
                    test_word = word[:i] + v + word[i:]
                    if test_word.lower() in self.term_dict:
                        return self.term_dict[test_word.lower()]

        # Too many consecutive consonants
        max_consonant_streak = 0
        current_streak = 0
        for c in word.lower():
            if c in consonants:
                current_streak += 1
                max_consonant_streak = max(max_consonant_streak, current_streak)
            else:
                current_streak = 0

        if max_consonant_streak > 3:
            # Try common substitution patterns
            substitutions = [
                ('c', 'a'), ('c', 'e'), ('c', 'o'),  # 'c' often substitutes vowels
                ('v', 'h'), ('t', 'h'), ('f', 'f'),  # Common substitutions
                ('i', 'l'), ('l', 'i'), ('n', 'm')  # Similar looking characters
            ]

            for old, new in substitutions:
                if old in word.lower():
                    test_word = word.lower().replace(old, new)
                    if test_word in self.term_dict:
                        return self.term_dict[test_word]

        return None
    def predict_from_context(self, tokens, target_idx, window_size=3):
        """Predict word based on surrounding context"""
        # Get context window
        start = max(0, target_idx - window_size)
        end = min(len(tokens), target_idx + window_size + 1)
        context = tokens[start:target_idx] + tokens[target_idx + 1:end]

        # Look for matching contexts in rules
        best_candidates = []
        best_score = 0

        for term, rule in self.context_rules.items():
            # Count context matches
            matches = sum(1 for word in context if word in rule["context_words"])
            score = matches / max(1, len(rule["context_words"]))

            if score > best_score:
                best_score = score
                best_candidates = rule["candidates"]

        if best_score > 0.5 and best_candidates:
            return best_candidates[0], best_score

        return None, 0

    def enhance_embedding(self, embedding, text):
        """Enhance embedding with knowledge base information"""
        # Query KB for relevant knowledge
        kb_results = self.query(text)

        # If no relevant knowledge, return original
        if not kb_results:
            return embedding

        # Simple enhancement: adjust dimensions based on term importance
        # In a real implementation, this would be more sophisticated
        enhanced = embedding.copy()
        importance_factor = min(1.0, len(kb_results) * 0.1)

        # Boost certain dimensions based on matched terms
        # (This is a simplified approach - a real system would have a more principled method)
        seed = sum(ord(c) for c in text) % 100  # Create a deterministic seed
        np.random.seed(seed)
        boost_indices = np.random.choice(len(enhanced), size=int(len(enhanced) * 0.1), replace=False)
        enhanced[boost_indices] *= (1.0 + importance_factor)

        # Normalize
        norm = np.linalg.norm(enhanced)
        if norm > 0:
            enhanced = enhanced / norm * np.linalg.norm(embedding)

        return enhanced

    def kb_guided_reconstruction(self, noisy_text):
        """Reconstruct text using knowledge base guidance with multiple correction strategies"""
        # Stage 1: Direct dictionary lookup
        words = noisy_text.split()
        corrected_words = []
        changes_made = False

        for word in words:
            # Try exact match in dictionary
            if word.lower() in self.term_dict:
                # Preserve capitalization
                if word[0].isupper() and len(self.term_dict[word.lower()]) > 0:
                    corrected = self.term_dict[word.lower()].capitalize()
                else:
                    corrected = self.term_dict[word.lower()]
                corrected_words.append(corrected)
                changes_made = True
                continue

            # Keep original for now - we'll do fuzzy matching in stage 2
            corrected_words.append(word)

        # Check if direct corrections were made
        if changes_made:
            return " ".join(corrected_words)

        # Stage 2: Apply fuzzy matching for words not directly found
        corrected_words = []
        changes_made = False

        for word in words:
            # Skip very short words and punctuation
            if len(word) <= 2 or all(c in '.,;:!?()[]{}"\'' for c in word):
                corrected_words.append(word)
                continue

            # Try fuzzy matching with a lower threshold
            best_match, score = self._fuzzy_match(word, threshold=0.7)
            if best_match and score > 0.7:
                # Preserve capitalization
                if word[0].isupper() and len(best_match) > 0:
                    corrected = best_match.capitalize()
                else:
                    corrected = best_match
                corrected_words.append(corrected)
                changes_made = True
                continue

            # Stage 3: Try pattern-based correction
            correction = self._pattern_based_correction(word)
            if correction:
                corrected_words.append(correction)
                changes_made = True
                continue

            # Keep original if no correction found
            corrected_words.append(word)

        # Final result
        return " ".join(corrected_words)

    def initialize_embedding_mapper(self, embeddings, texts):
        """Initialize text-embedding mapper with reference data"""
        if not hasattr(self, 'kb_mapper') or self.kb_mapper is None:
            self.kb_mapper = TextEmbeddingMapper()

        # Fit mapper with provided data
        self.kb_mapper.fit(embeddings, texts)
        logger.info(f"Initialized KB embedding mapper with {len(embeddings)} examples")
        return True
    def get_importance_weights(self, embedding):
        """Get importance weights for embedding dimensions based on KB knowledge"""
        # Default implementation - more sophisticated in practice
        weights = np.ones_like(embedding)

        # Try to map embedding to text for better weighting
        if hasattr(self, 'kb_mapper') and self.kb_mapper is not None:
            try:
                text = self.kb_mapper.embedding_to_text(embedding)
                # Get active terms
                terms = [word for word in text.split() if word.lower() in self.term_dict]

                if terms:
                    # Enhance weights for dimensions associated with known terms
                    importance_factor = min(0.5, len(terms) / len(text.split()) * 0.8)

                    # Use a deterministic seed for reproducibility
                    seed = sum(ord(c) for c in ''.join(terms)) % 1000
                    np.random.seed(seed)

                    # Select dimensions to enhance (about 20% of dimensions)
                    enhance_dims = np.random.choice(
                        len(weights),
                        size=int(0.2 * len(weights)),
                        replace=False
                    )

                    # Boost importance of these dimensions
                    weights[enhance_dims] += importance_factor

                    # Normalize
                    weights = weights / np.sum(weights) * len(weights)
            except Exception as e:
                logger.debug(f"Error generating importance weights: {e}")

        return weights
    def enhance_with_context(self, text, context):
        """Enhance text reconstruction using surrounding context"""
        import difflib

        # Tokenize both
        text_tokens = text.split()
        context_tokens = context.split()

        # Find phrases that appear in context
        context_phrases = []
        for i in range(len(context_tokens) - 1):
            for j in range(i + 1, min(i + 5, len(context_tokens))):
                phrase = ' '.join(context_tokens[i:j + 1])
                if len(phrase.split()) > 1:  # Only phrases with 2+ words
                    context_phrases.append(phrase)

        # Check if any part of the text needs enhancement
        replacements = []
        for i in range(len(text_tokens)):
            for j in range(i + 1, min(i + 5, len(text_tokens))):
                text_phrase = ' '.join(text_tokens[i:j + 1])

                # Find closest match in context phrases
                best_match = None
                best_score = 0

                for context_phrase in context_phrases:
                    score = difflib.SequenceMatcher(None, text_phrase.lower(), context_phrase.lower()).ratio()
                    if score > 0.7 and score > best_score:  # 70% similarity threshold
                        best_score = score
                        best_match = context_phrase

                if best_match and best_score > 0.7:
                    replacements.append((i, j, best_match))

        # Apply replacements (start from the end to maintain indices)
        replacements.sort(reverse=True)
        for start, end, replacement in replacements:
            # Replace slice with context-matched phrase
            text_tokens[start:end + 1] = replacement.split()

        return ' '.join(text_tokens)
def get_or_create_knowledge_base(domain="europarl", kb_path=None):
    """Get or create a knowledge base for the specified domain"""
    return SemanticKnowledgeBase(domain=domain, kb_path=kb_path)


def test_kb_integration():
    """Test the knowledge base integration"""
    print("=== Testing Knowledge Base Integration ===")

    # Create knowledge base
    kb = get_or_create_knowledge_base()

    # Test correction
    test_sentence = "The Parliamemt will now vote on the propofal from the Commissiob."
    corrected = kb.kb_guided_reconstruction(test_sentence)
    print(f"Original: {test_sentence}")
    print(f"Corrected: {corrected}")

    # Test embedding enhancement
    import numpy as np
    test_embedding = np.random.rand(768)
    enhanced = kb.enhance_embedding(test_embedding, test_sentence)
    similarity = np.dot(test_embedding, enhanced) / (np.linalg.norm(test_embedding) * np.linalg.norm(enhanced))
    print(f"Embedding enhancement similarity: {similarity:.4f}")

    print("=== Test Complete ===")


if __name__ == "__main__":
    test_kb_integration()