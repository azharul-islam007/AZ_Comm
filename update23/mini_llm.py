# mini_llm.py
import torch
import os
import logging
import difflib
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class MiniLLM:
    """Lightweight local LLM for text reconstruction as an API fallback"""

    def __init__(self, model_name="distilgpt2", device=None):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialized = False
        self.load_model()

    def load_model(self):
        """Load the model and tokenizer"""
        try:
            logger.info(f"Loading Mini-LLM model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

            # Fix for padding token - set pad_token to eos_token if not defined
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info(f"Set padding token to EOS token: {self.tokenizer.eos_token}")

            self.model.to(self.device)
            self.model.eval()
            self.initialized = True
            logger.info(f"Mini-LLM initialized successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize Mini-LLM: {e}")
            self.initialized = False

    def estimate_confidence(self, noisy_text, corrected_text):
        """Estimate confidence in the generated correction"""
        # Simple initial confidence based on string similarity
        similarity = difflib.SequenceMatcher(None, noisy_text, corrected_text).ratio()

        # Adjust confidence based on whether common parliamentary terms are preserved
        parl_terms = ["parliament", "commission", "council", "rule", "directive", "regulation"]
        term_preservation = 0
        total_terms = 0

        for term in parl_terms:
            if term in noisy_text.lower():
                total_terms += 1
                if term in corrected_text.lower():
                    term_preservation += 1

        # If parliamentary terms present, factor their preservation into confidence
        if total_terms > 0:
            term_factor = term_preservation / total_terms
            confidence = 0.7 * similarity + 0.3 * term_factor
        else:
            confidence = similarity

        return min(0.95, confidence)  # Cap at 0.95 to leave room for improvement

    def reconstruct(self, noisy_text, context=None, min_confidence=0.6):
        """Reconstruct text using the mini-LLM"""
        if not self.initialized:
            logger.warning("Mini-LLM not initialized")
            return noisy_text, 0.0

        try:
            # Prepare prompt
            if context:
                prompt = f"Context: {context}\n\nCorrect this text: {noisy_text}\n\nCorrected:"
            else:
                prompt = f"Correct this text: {noisy_text}\n\nCorrected:"

            # Tokenize with explicit padding and attention mask
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            attention_mask = inputs.attention_mask.to(self.device)
            inputs = inputs.to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=len(noisy_text.split()) * 2,
                    do_sample=True,  # Enable sampling since we're using temperature/top_p
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode and extract corrected text
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            corrected_text = full_output.split("Corrected:")[-1].strip()

            # Estimate confidence
            confidence = self.estimate_confidence(noisy_text, corrected_text)

            # Only return if confidence exceeds threshold
            if confidence >= min_confidence:
                return corrected_text, confidence
            else:
                return noisy_text, confidence

        except Exception as e:
            logger.error(f"Mini-LLM reconstruction error: {e}")
            return noisy_text, 0.0