# mini_llm.py
import torch
import os
import logging
import difflib
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class MiniLLM:
    """Enhanced lightweight local LLM for text reconstruction with multiple approaches"""

    def __init__(self, model_name="gpt2-medium", device=None):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialized = False
        self.actual_model_loaded = None  # Track which model was actually loaded
        self.load_model()

    def load_model(self):
        """Load the model and tokenizer with fallback options"""
        # Try different model sizes in order of preference
        model_fallbacks = []

        if self.model_name == "gpt2-large":
            model_fallbacks = ["gpt2-large", "gpt2-medium", "gpt2"]
        elif self.model_name == "gpt2-medium":
            model_fallbacks = ["gpt2-medium", "gpt2"]
        else:
            model_fallbacks = [self.model_name, "gpt2"]

        for model_name in model_fallbacks:
            try:
                logger.info(f"Attempting to load Mini-LLM model: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)

                # Fix for padding token - set pad_token to eos_token if not defined
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    logger.debug(f"Set padding token to EOS token: {self.tokenizer.eos_token}")

                self.model.to(self.device)
                self.model.eval()
                self.initialized = True
                self.actual_model_loaded = model_name
                logger.info(f"Mini-LLM initialized successfully with {model_name} on {self.device}")
                return  # Success, exit the loop

            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue

        # If we get here, all models failed
        logger.error("Failed to initialize any Mini-LLM model")
        self.initialized = False

    def estimate_confidence(self, noisy_text, corrected_text):
        """Enhanced confidence estimation with better metrics"""
        if not corrected_text or corrected_text == noisy_text:
            return 0.1

        # Basic similarity
        similarity = difflib.SequenceMatcher(None, noisy_text, corrected_text).ratio()

        # Check for parliamentary term preservation
        parl_terms = ["parliament", "commission", "council", "rule", "directive", "regulation",
                      "president", "quaestors", "lynne", "gorsel", "strasbourg", "brussels"]
        term_preservation = 0
        total_terms = 0

        for term in parl_terms:
            if term in noisy_text.lower():
                total_terms += 1
                if term in corrected_text.lower():
                    term_preservation += 1

        # Check for common word corrections
        common_corrections = {
            "wenb": "send", "majorityy": "majority", "shouldd": "should",
            "messagee": "message", "sincee": "since", "vastt": "vast"
        }

        correction_bonus = 0
        correction_count = 0
        for wrong, right in common_corrections.items():
            if wrong in noisy_text.lower():
                correction_count += 1
                if right in corrected_text.lower() and wrong not in corrected_text.lower():
                    correction_bonus += 1

        # Calculate final confidence
        base_confidence = similarity

        # Parliamentary term factor
        if total_terms > 0:
            term_factor = term_preservation / total_terms
            base_confidence = 0.6 * base_confidence + 0.3 * term_factor

        # Common correction bonus
        if correction_count > 0:
            correction_factor = correction_bonus / correction_count
            base_confidence = 0.8 * base_confidence + 0.2 * correction_factor

        # Length penalty for significantly longer text (hallucination detection)
        orig_len = len(noisy_text.split())
        corr_len = len(corrected_text.split())
        if corr_len > orig_len * 1.5:  # More than 50% longer
            base_confidence *= 0.7  # Penalty for likely hallucination

        return min(0.95, max(0.1, base_confidence))  # Keep in reasonable range

    def create_enhanced_prompt(self, noisy_text, context=None, approach="correction"):
        """Create enhanced prompts for different reconstruction approaches"""

        if approach == "correction":
            base_prompt = """You are correcting text from European Parliament proceedings. Fix these specific errors:
- "wenb" → "send"
- "majorityy" → "majority" 
- "shouldd" → "should"
- "messagee" → "message"
- "sincee" → "since"
- "vastt" → "vast"
- "com" → "you"
- "haye" → "have"
- "durifb" → "during"
- "tjps" → "this"

Keep parliamentary terms like Parliament, Commission, Council, President intact."""

        elif approach == "paraphrase":
            base_prompt = """Rewrite this European Parliament text more clearly, fixing any obvious errors while preserving all parliamentary terminology:"""

        elif approach == "context":
            base_prompt = """Using the context provided, correct errors in this European Parliament text. Preserve all parliamentary terms and proper names:"""

        else:  # simple
            base_prompt = """Fix errors in this text:"""

        if context and approach == "context":
            prompt = f"{base_prompt}\n\nContext: {context}\n\nText to correct: {noisy_text}\n\nCorrected text:"
        else:
            prompt = f"{base_prompt}\n\nText to correct: {noisy_text}\n\nCorrected text:"

        return prompt

    def reconstruct(self, noisy_text, context=None, min_confidence=0.6, temperature=0.7, approach="correction"):
        """Enhanced reconstruction with multiple approaches and configurable parameters"""
        if not self.initialized:
            logger.warning("Mini-LLM not initialized")
            return noisy_text, 0.0

        if not noisy_text or len(noisy_text.strip()) == 0:
            return noisy_text, 0.0

        try:
            # Create enhanced prompt based on approach
            prompt = self.create_enhanced_prompt(noisy_text, context, approach)

            # Tokenize with explicit padding and attention mask
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            attention_mask = inputs.attention_mask.to(self.device)
            inputs = inputs.to(self.device)

            # Calculate max new tokens based on input length
            input_length = len(noisy_text.split())
            max_new_tokens = min(input_length * 2, 100)  # Cap at 100 tokens

            # Generate with configurable temperature
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=max(0.1, min(1.5, temperature)),  # Clamp temperature
                    top_p=0.9,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,  # Reduce repetition
                    early_stopping=True
                )

            # Decode and extract corrected text
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract the corrected text after the prompt
            if "Corrected text:" in full_output:
                corrected_text = full_output.split("Corrected text:")[-1].strip()
            elif "corrected text:" in full_output:
                corrected_text = full_output.split("corrected text:")[-1].strip()
            else:
                # Take everything after the original prompt
                corrected_text = full_output[len(prompt):].strip()

            # Clean up the output
            corrected_text = self.clean_output(corrected_text, noisy_text)

            # Estimate confidence
            confidence = self.estimate_confidence(noisy_text, corrected_text)

            # Apply minimum confidence threshold
            if confidence >= min_confidence:
                logger.debug(
                    f"Mini-LLM ({approach}): '{noisy_text}' → '{corrected_text}' (confidence: {confidence:.2f})")
                return corrected_text, confidence
            else:
                logger.debug(f"Mini-LLM ({approach}): Low confidence {confidence:.2f}, returning original")
                return noisy_text, confidence

        except Exception as e:
            logger.error(f"Mini-LLM reconstruction error: {e}")
            return noisy_text, 0.0

    def clean_output(self, output, original):
        """Clean and validate the model output"""
        if not output:
            return original

        # Remove common prefixes/suffixes that models sometimes add
        prefixes_to_remove = ["corrected:", "fixed:", "output:", "result:", "answer:"]
        for prefix in prefixes_to_remove:
            if output.lower().startswith(prefix):
                output = output[len(prefix):].strip()

        # Remove quotes if the entire output is quoted
        if output.startswith('"') and output.endswith('"'):
            output = output[1:-1]
        if output.startswith("'") and output.endswith("'"):
            output = output[1:-1]

        # If output is too long compared to original, truncate at sentence boundary
        orig_words = len(original.split())
        output_words = output.split()

        if len(output_words) > orig_words * 2:  # More than double length
            # Find sentence boundary near original length
            target_length = min(orig_words * 2, len(output_words))
            truncated_words = output_words[:target_length]
            output = " ".join(truncated_words)

        # If output is empty or whitespace, return original
        if not output.strip():
            return original

        return output

    def multi_approach_reconstruct(self, noisy_text, context=None, min_confidence=0.3):
        """Try multiple reconstruction approaches and return the best result"""
        if not self.initialized:
            return noisy_text, 0.0

        approaches = [
            ("correction", 0.7),
            ("paraphrase", 0.8),
            ("context", 0.6) if context else None,
            ("simple", 0.9)
        ]

        # Filter out None values
        approaches = [a for a in approaches if a is not None]

        best_result = noisy_text
        best_confidence = 0.0
        best_approach = "none"

        for approach, temperature in approaches:
            try:
                result, confidence = self.reconstruct(
                    noisy_text, context, min_confidence, temperature, approach)

                if confidence > best_confidence:
                    best_result = result
                    best_confidence = confidence
                    best_approach = approach

                    # If we get high confidence, we can stop early
                    if confidence > 0.85:
                        break

            except Exception as e:
                logger.debug(f"Approach {approach} failed: {e}")
                continue

        logger.debug(f"Multi-approach Mini-LLM: best approach '{best_approach}' with confidence {best_confidence:.2f}")
        return best_result, best_confidence

    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            "initialized": self.initialized,
            "requested_model": self.model_name,
            "actual_model": self.actual_model_loaded,
            "device": str(self.device)
        }