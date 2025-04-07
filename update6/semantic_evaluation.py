# semantic_evaluation.py
import numpy as np
import torch
import logging
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedEvaluationFramework:
    """
    Advanced evaluation framework for semantic communication systems
    that provides more comprehensive metrics beyond standard NLP measures.
    """

    def __init__(self, semantic_loss=None):
        """
        Initialize the evaluation framework

        Args:
            semantic_loss: Optional semantic loss module for similarity calculation
        """
        self.semantic_loss = semantic_loss

        # Try to initialize semantic loss if not provided
        if self.semantic_loss is None:
            try:
                from semantic_loss import get_semantic_loss
                self.semantic_loss = get_semantic_loss()
                logger.info("Initialized semantic loss for evaluation")
            except:
                logger.warning("Could not initialize semantic loss module")

        # Initialize ROUGE scorer
        try:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        except:
            logger.warning("Could not initialize ROUGE scorer")
            self.rouge_scorer = None

        # Prepare knowledge base for terminology evaluation
        self.kb = None
        try:
            from knowledge_base import get_or_create_knowledge_base
            self.kb = get_or_create_knowledge_base()
            logger.info("Initialized knowledge base for evaluation")
        except:
            logger.warning("Could not initialize knowledge base")

    def evaluate_reconstruction(self, original_texts, reconstructed_texts):
        """
        Comprehensive evaluation of reconstruction quality

        Args:
            original_texts: List of original texts or single text
            reconstructed_texts: List of reconstructed texts or single text

        Returns:
            Dictionary of evaluation metrics
        """
        # Handle single text input
        if isinstance(original_texts, str):
            original_texts = [original_texts]
        if isinstance(reconstructed_texts, str):
            reconstructed_texts = [reconstructed_texts]

        # Ensure same length
        min_len = min(len(original_texts), len(reconstructed_texts))
        original_texts = original_texts[:min_len]
        reconstructed_texts = reconstructed_texts[:min_len]

        # Storage for results
        results = {
            "standard_metrics": {},
            "semantic_metrics": {},
            "domain_metrics": {},
            "information_metrics": {}
        }

        # Calculate standard metrics
        standard_metrics = self._calculate_standard_metrics(original_texts, reconstructed_texts)
        results["standard_metrics"] = standard_metrics

        # Calculate semantic metrics
        semantic_metrics = self._calculate_semantic_metrics(original_texts, reconstructed_texts)
        results["semantic_metrics"] = semantic_metrics

        # Calculate domain-specific metrics
        domain_metrics = self._calculate_domain_metrics(original_texts, reconstructed_texts)
        results["domain_metrics"] = domain_metrics

        # Calculate information-theoretic metrics
        info_metrics = self._calculate_information_metrics(original_texts, reconstructed_texts)
        results["information_metrics"] = info_metrics

        # Calculate overall scores
        results["overall"] = self._calculate_overall_scores(results)

        return results

    def _calculate_standard_metrics(self, original_texts, reconstructed_texts):
        """Calculate standard NLP evaluation metrics"""
        metrics = {
            "bleu": [],
            "rouge1_precision": [],
            "rouge1_recall": [],
            "rouge1_f1": [],
            "rouge2_f1": [],
            "rougeL_f1": [],
            "meteor": []
        }

        # Process each pair
        for orig, recon in zip(original_texts, reconstructed_texts):
            # BLEU score
            try:
                bleu = sentence_bleu([orig.split()], recon.split(),
                                     smoothing_function=SmoothingFunction().method4)
                metrics["bleu"].append(bleu)
            except:
                metrics["bleu"].append(0.0)

            # ROUGE scores
            try:
                if self.rouge_scorer:
                    rouge = self.rouge_scorer.score(orig, recon)
                    metrics["rouge1_precision"].append(rouge["rouge1"].precision)
                    metrics["rouge1_recall"].append(rouge["rouge1"].recall)
                    metrics["rouge1_f1"].append(rouge["rouge1"].fmeasure)
                    metrics["rouge2_f1"].append(rouge["rouge2"].fmeasure)
                    metrics["rougeL_f1"].append(rouge["rougeL"].fmeasure)
                else:
                    metrics["rouge1_precision"].append(0.0)
                    metrics["rouge1_recall"].append(0.0)
                    metrics["rouge1_f1"].append(0.0)
                    metrics["rouge2_f1"].append(0.0)
                    metrics["rougeL_f1"].append(0.0)
            except:
                metrics["rouge1_precision"].append(0.0)
                metrics["rouge1_recall"].append(0.0)
                metrics["rouge1_f1"].append(0.0)
                metrics["rouge2_f1"].append(0.0)
                metrics["rougeL_f1"].append(0.0)

            # METEOR score
            try:
                meteor = meteor_score([orig.split()], recon.split())
                metrics["meteor"].append(meteor)
            except:
                metrics["meteor"].append(0.0)

        # Calculate averages
        for key in metrics:
            if metrics[key]:
                metrics[f"{key}_avg"] = sum(metrics[key]) / len(metrics[key])
            else:
                metrics[f"{key}_avg"] = 0.0

        return metrics

    def _calculate_semantic_metrics(self, original_texts, reconstructed_texts):
        """Calculate semantic similarity metrics"""
        metrics = {
            "semantic_similarity": [],
            "context_preservation": [],
            "logic_flow": []
        }

        # Process each pair
        for orig, recon in zip(original_texts, reconstructed_texts):
            # Semantic similarity via semantic loss
            if self.semantic_loss:
                try:
                    similarity = self.semantic_loss.calculate_semantic_similarity(orig, recon)
                    metrics["semantic_similarity"].append(similarity)
                except:
                    metrics["semantic_similarity"].append(0.0)
            else:
                # Fallback approximation
                metrics["semantic_similarity"].append(
                    self._approx_semantic_similarity(orig, recon))

            # Context preservation - measure if key context words are preserved
            context_score = self._measure_context_preservation(orig, recon)
            metrics["context_preservation"].append(context_score)

            # Logic flow - evaluate if logical structure is maintained
            logic_score = self._evaluate_logic_flow(orig, recon)
            metrics["logic_flow"].append(logic_score)

        # Calculate averages
        for key in metrics:
            if metrics[key]:
                metrics[f"{key}_avg"] = sum(metrics[key]) / len(metrics[key])
            else:
                metrics[f"{key}_avg"] = 0.0

        return metrics

    def _calculate_domain_metrics(self, original_texts, reconstructed_texts):
        """Calculate domain-specific metrics (Europarl domain)"""
        metrics = {
            "terminology_preservation": [],
            "procedural_accuracy": [],
            "entity_preservation": []
        }

        # Get domain-specific terms
        domain_terms = set()
        if self.kb and hasattr(self.kb, 'term_dict'):
            domain_terms = set(self.kb.term_dict.values())

        # Europarl-specific important terms if KB doesn't have them
        if not domain_terms:
            domain_terms = {
                "Parliament", "Commission", "Council", "Directive", "Regulation",
                "Committee", "Member", "State", "States", "European", "Union",
                "vote", "voting", "proposal", "amendment", "debate", "session",
                "agenda", "Rule", "procedure", "codecision", "legislation"
            }

        # Process each pair
        for orig, recon in zip(original_texts, reconstructed_texts):
            # Terminology preservation - check if domain terms are preserved
            orig_terms = set(w for w in orig.split() if w in domain_terms or w.strip(",.;:()[]{}\"'") in domain_terms)
            recon_terms = set(w for w in recon.split() if w in domain_terms or w.strip(",.;:()[]{}\"'") in domain_terms)

            if orig_terms:
                term_score = len(orig_terms.intersection(recon_terms)) / len(orig_terms)
            else:
                term_score = 1.0  # No terms to preserve

            metrics["terminology_preservation"].append(term_score)

            # Procedural accuracy - check for procedural phrases
            proc_score = self._evaluate_procedural_content(orig, recon)
            metrics["procedural_accuracy"].append(proc_score)

            # Entity preservation - named entities, numbers, dates
            entity_score = self._evaluate_entity_preservation(orig, recon)
            metrics["entity_preservation"].append(entity_score)

        # Calculate averages
        for key in metrics:
            if metrics[key]:
                metrics[f"{key}_avg"] = sum(metrics[key]) / len(metrics[key])
            else:
                metrics[f"{key}_avg"] = 0.0

        return metrics

    def _calculate_information_metrics(self, original_texts, reconstructed_texts):
        """Calculate information-theoretic metrics"""
        metrics = {
            "lexical_diversity_ratio": [],
            "length_preservation": [],
            "key_info_preservation": []
        }

        # Process each pair
        for orig, recon in zip(original_texts, reconstructed_texts):
            # Lexical diversity ratio - vocabulary richness comparison
            orig_diversity = len(set(orig.lower().split())) / max(1, len(orig.split()))
            recon_diversity = len(set(recon.lower().split())) / max(1, len(recon.split()))

            if orig_diversity > 0:
                diversity_ratio = min(1.0, recon_diversity / orig_diversity)
            else:
                diversity_ratio = 1.0

            metrics["lexical_diversity_ratio"].append(diversity_ratio)

            # Length preservation
            orig_len = len(orig.split())
            recon_len = len(recon.split())

            if orig_len > 0:
                length_ratio = min(1.0, recon_len / orig_len)
            else:
                length_ratio = 1.0

            metrics["length_preservation"].append(length_ratio)

            # Key information preservation - focuses on important words
            key_info_score = self._evaluate_key_info_preservation(orig, recon)
            metrics["key_info_preservation"].append(key_info_score)

        # Calculate averages
        for key in metrics:
            if metrics[key]:
                metrics[f"{key}_avg"] = sum(metrics[key]) / len(metrics[key])
            else:
                metrics[f"{key}_avg"] = 0.0

        return metrics

    def _calculate_overall_scores(self, results):
        """Calculate overall scores based on all metrics"""
        overall = {}

        # Semantic fidelity score - weighted combination of semantic metrics
        semantic_metrics = results["semantic_metrics"]
        semantic_fidelity = (
                0.6 * semantic_metrics.get("semantic_similarity_avg", 0.0) +
                0.2 * semantic_metrics.get("context_preservation_avg", 0.0) +
                0.2 * semantic_metrics.get("logic_flow_avg", 0.0)
        )
        overall["semantic_fidelity"] = semantic_fidelity

        # Linguistic quality score - based on standard metrics
        standard_metrics = results["standard_metrics"]
        linguistic_quality = (
                0.3 * standard_metrics.get("bleu_avg", 0.0) +
                0.3 * standard_metrics.get("rougeL_f1_avg", 0.0) +
                0.4 * standard_metrics.get("meteor_avg", 0.0)
        )
        overall["linguistic_quality"] = linguistic_quality

        # Domain relevance score - based on domain-specific metrics
        domain_metrics = results["domain_metrics"]
        domain_relevance = (
                0.5 * domain_metrics.get("terminology_preservation_avg", 0.0) +
                0.3 * domain_metrics.get("procedural_accuracy_avg", 0.0) +
                0.2 * domain_metrics.get("entity_preservation_avg", 0.0)
        )
        overall["domain_relevance"] = domain_relevance

        # Information preservation score
        info_metrics = results["information_metrics"]
        info_preservation = (
                0.3 * info_metrics.get("lexical_diversity_ratio_avg", 0.0) +
                0.2 * info_metrics.get("length_preservation_avg", 0.0) +
                0.5 * info_metrics.get("key_info_preservation_avg", 0.0)
        )
        overall["information_preservation"] = info_preservation

        # Combined overall score
        overall["overall_score"] = (
                0.4 * semantic_fidelity +
                0.3 * linguistic_quality +
                0.2 * domain_relevance +
                0.1 * info_preservation
        )

        return overall

    # Helper methods for specific metric calculations
    def _approx_semantic_similarity(self, text1, text2):
        """Approximate semantic similarity when semantic loss isn't available"""
        # Count word overlap (removing stop words)
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
        words1 = [w.lower() for w in text1.split() if w.lower() not in stop_words]
        words2 = [w.lower() for w in text2.split() if w.lower() not in stop_words]

        # Count matching words
        matches = sum(1 for w in words1 if w in words2)

        if not words1:
            return 0.0

        # Calculate similarity
        return min(1.0, matches / max(1, len(words1)))

    def _measure_context_preservation(self, text1, text2):
        """Measure preservation of context words"""
        # Focus on content words (nouns, verbs, adjectives)
        content_words1 = set(w.lower() for w in text1.split() if len(w) > 3)
        content_words2 = set(w.lower() for w in text2.split() if len(w) > 3)

        if not content_words1:
            return 1.0

        # Calculate overlap
        overlap = len(content_words1.intersection(content_words2))
        return min(1.0, overlap / max(1, len(content_words1)))

    def _evaluate_logic_flow(self, text1, text2):
        """Evaluate preservation of logical flow and structure"""
        # Check for logical connectors
        logical_connectors = {'if', 'then', 'because', 'therefore', 'since', 'although',
                              'however', 'but', 'and', 'or', 'so'}

        conn1 = [w.lower() for w in text1.split() if w.lower() in logical_connectors]
        conn2 = [w.lower() for w in text2.split() if w.lower() in logical_connectors]

        # Connector score
        if conn1:
            conn_score = len(set(conn1).intersection(set(conn2))) / len(conn1)
        else:
            conn_score = 1.0

        # Sentence structure similarity (approximation)
        structure_score = self._evaluate_sentence_structure(text1, text2)

        # Combine scores
        return 0.6 * conn_score + 0.4 * structure_score

    def _evaluate_sentence_structure(self, text1, text2):
        """Evaluate similarity of sentence structure"""
        # Simplistic approach: check if sentence lengths are similar
        len1 = len(text1.split())
        len2 = len(text2.split())

        if len1 == 0:
            return 1.0

        length_ratio = min(len1, len2) / max(len1, len2)

        # Check if punctuation patterns are similar
        punct1 = [c for c in text1 if c in ',.;:?!()']
        punct2 = [c for c in text2 if c in ',.;:?!()']

        if punct1:
            punct_ratio = min(len(punct1), len(punct2)) / max(len(punct1), len(punct2))
        else:
            punct_ratio = 1.0 if not punct2 else 0.5

        return 0.7 * length_ratio + 0.3 * punct_ratio

    def _evaluate_procedural_content(self, text1, text2):
        """Evaluate preservation of procedural content (Europarl specific)"""
        # Procedural phrases common in Europarl
        procedural_phrases = {
            'in accordance with', 'pursuant to', 'rule', 'article', 'on the agenda',
            'the session', 'the meeting', 'the vote', 'voting', 'motion', 'proposal'
        }

        # Check for presence of procedural phrases
        proc1 = sum(1 for phrase in procedural_phrases if phrase in text1.lower())
        proc2 = sum(1 for phrase in procedural_phrases if phrase in text2.lower())

        if proc1 > 0:
            return min(1.0, proc2 / proc1)
        else:
            return 1.0 if proc2 == 0 else 0.8

    def _evaluate_entity_preservation(self, text1, text2):
        """Evaluate preservation of named entities, numbers, and dates"""
        # Count capitalized words (potential named entities)
        caps1 = [w for w in text1.split() if w and w[0].isupper()]
        caps2 = [w for w in text2.split() if w and w[0].isupper()]

        # Entity score
        if caps1:
            entity_score = len(set(caps1).intersection(set(caps2))) / len(caps1)
        else:
            entity_score = 1.0 if not caps2 else 0.8

        # Count numbers and dates (simple approach)
        nums1 = [w for w in text1.split() if any(c.isdigit() for c in w)]
        nums2 = [w for w in text2.split() if any(c.isdigit() for c in w)]

        # Number score
        if nums1:
            num_score = len(set(nums1).intersection(set(nums2))) / len(nums1)
        else:
            num_score = 1.0 if not nums2 else 0.8

        return 0.6 * entity_score + 0.4 * num_score

    def _evaluate_key_info_preservation(self, text1, text2):
        """Evaluate preservation of key information words"""
        # Identify potential key information words (content words)
        # Exclude stopwords and short words
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}

        # Extract content words (longer words more likely to be important)
        content1 = [w.lower() for w in text1.split() if w.lower() not in stop_words and len(w) > 4]
        content2 = [w.lower() for w in text2.split() if w.lower() not in stop_words and len(w) > 4]

        if not content1:
            return 1.0

        # Calculate score based on overlap of content words
        overlap = len(set(content1).intersection(set(content2)))
        return min(1.0, overlap / len(content1))