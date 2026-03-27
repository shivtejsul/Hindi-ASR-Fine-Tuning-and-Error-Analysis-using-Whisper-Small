"""
Question 4: Lattice-based WER Evaluation
=========================================

Tasks:
a) Design lattice structure capturing valid transcription alternatives
b) Handle insertions, deletions, substitutions fairly
c) Decide when to trust model agreement over reference
d) Compute WER using lattice-based method

Key Insight:
Multiple valid representations should exist for same audio.
Examples:
- Numbers: चौदह vs 14
- Synonyms: कताबें vs पुस्तकें vs किताब
- Spelling: खरीदीं vs खरीदी
"""

from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
import numpy as np
from jiwer import wer, cer
import logging
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlignmentUnit(Enum):
    """Units for alignment and WER calculation"""
    WORD = "word"
    SUBWORD = "subword"
    CHARACTER = "character"
    PHRASE = "phrase"


@dataclass
class LatticeNode:
    """Node in a lattice representing alignment position"""
    position: int  # Position in audio timeline
    alternatives: List[str] = field(default_factory=list)  # Valid alternatives at this position
    
    def add_alternative(self, text: str):
        """Add a valid alternative representation"""
        if text not in self.alternatives:
            self.alternatives.append(text)
    
    def is_empty(self) -> bool:
        """Check if node has no alternatives"""
        return len(self.alternatives) == 0


@dataclass
class Lattice:
    """
    Lattice structure capturing multiple valid transcriptions
    
    Example:
    Spoken: "उसने चौदह कताबें खरीदीं"
    Lattice:
    [
        [["उसने"]],
        [["चौदह", "14"]],
        [["कताबें", "कताबे", "पुस्तकें"]],
        [["खरीदीं", "खरीदी"]]
    ]
    """
    
    nodes: List[LatticeNode] = field(default_factory=list)
    reference_transcript: str = ""
    model_outputs: List[str] = field(default_factory=list)
    
    def add_node(self, position: int, alternatives: List[str]):
        """Add a node with alternatives"""
        node = LatticeNode(position=position, alternatives=alternatives)
        self.nodes.append(node)
    
    def get_node(self, position: int) -> Optional[LatticeNode]:
        """Get node at position"""
        for node in self.nodes:
            if node.position == position:
                return node
        return None
    
    def to_sequence_of_bins(self) -> List[List[str]]:
        """Convert lattice to sequence of bins"""
        return [node.alternatives for node in sorted(self.nodes, key=lambda n: n.position)]


class LatticeBuilder:
    """Build lattice from multiple model outputs and reference"""
    
    def __init__(self, alignment_unit: AlignmentUnit = AlignmentUnit.WORD):
        self.alignment_unit = alignment_unit
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize based on alignment unit"""
        text = text.strip()
        
        if self.alignment_unit == AlignmentUnit.WORD:
            return text.split()
        elif self.alignment_unit == AlignmentUnit.CHARACTER:
            return list(text)
        elif self.alignment_unit == AlignmentUnit.SUBWORD:
            # Would use a subword tokenizer
            return text.split()
        elif self.alignment_unit == AlignmentUnit.PHRASE:
            # Split on punctuation
            import re
            return re.split(r'[।,।!?॥]', text)
        
        return text.split()
    
    def align_sequences(self, reference_tokens: List[str], 
                       predicted_tokens: List[str]) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Align two token sequences using edit distance
        
        Returns:
            List of alignment pairs: (ref_token, pred_token)
        """
        from difflib import SequenceMatcher
        
        matcher = SequenceMatcher(None, reference_tokens, predicted_tokens)
        matching_blocks = matcher.get_matching_blocks()
        
        aligned = []
        for block in matching_blocks:
            for i in range(block.size):
                ref_idx = block.a + i
                pred_idx = block.b + i
                aligned.append((reference_tokens[ref_idx], predicted_tokens[pred_idx]))
        
        return aligned
    
    def build_lattice(self, reference: str, model_outputs: List[str],
                     consensus_threshold: float = 0.5) -> Lattice:
        """
        Build lattice from reference and model outputs
        
        Args:
            reference: Human reference transcription
            model_outputs: List of model output transcriptions
            consensus_threshold: When model agreement > threshold, trust over reference
        
        Returns:
            Lattice with valid alternatives
        """
        lattice = Lattice(reference_transcript=reference, model_outputs=model_outputs)
        
        # Tokenize all texts
        ref_tokens = self.tokenize(reference)
        model_token_lists = [self.tokenize(output) for output in model_outputs]
        
        # Find consensus length
        max_len = max(len(ref_tokens), max(len(mt) for mt in model_token_lists))
        
        # For each position, collect valid alternatives
        for position in range(max_len):
            alternatives = set()
            
            # Add reference token if exists
            if position < len(ref_tokens):
                alternatives.add(ref_tokens[position])
            
            # Collect tokens from models at this position
            model_tokens_at_pos = []
            for model_tokens in model_token_lists:
                if position < len(model_tokens):
                    model_tokens_at_pos.append(model_tokens[position])
            
            # Add model tokens if there's agreement
            token_counts = {}
            for token in model_tokens_at_pos:
                token_counts[token] = token_counts.get(token, 0) + 1
            
            # Add tokens with high agreement
            for token, count in token_counts.items():
                agreement_ratio = count / len(model_outputs)
                if agreement_ratio >= consensus_threshold:
                    alternatives.add(token)
            
            # Also add any model variants (not just high agreement)
            for token in model_tokens_at_pos:
                alternatives.add(token)
            
            if alternatives:
                lattice.add_node(position, list(alternatives))
        
        return lattice
    
    def should_trust_models_over_reference(self, reference_token: str,
                                          model_tokens: List[str],
                                          consensus_threshold: float = 0.7) -> bool:
        """
        Decide when to trust model agreement over reference
        
        Rules:
        - If N models agree on same token (N > threshold), trust agreement
        - If reference is obviously wrong (e.g., all caps, garbled), trust models
        - If models disagree, keep reference unless strong evidence against it
        """
        if not model_tokens:
            return False
        
        # Count agreement
        token_counts = {}
        for token in model_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        # Check if most models agree on same token (different from reference)
        for token, count in token_counts.items():
            agreement = count / len(model_tokens)
            if agreement >= consensus_threshold and token != reference_token:
                # Check if reference seems suspicious
                if self._seems_erroneous(reference_token):
                    return True
        
        return False
    
    def _seems_erroneous(self, token: str) -> bool:
        """
        Heuristics to detect obviously erroneous tokens
        """
        # Check for unusual character patterns
        if len(token) > 50:  # Suspiciously long
            return True
        
        # Check if all uppercase (might be noise)
        if token.isupper():
            return True
        
        # Check for excessive repetition
        if len(set(token)) < len(token) * 0.3:  # < 30% unique chars
            return True
        
        return False


class LatticeLatticeWERCalculator:
    """Calculate WER using lattice-based evaluation"""
    
    def __init__(self, alignment_unit: AlignmentUnit = AlignmentUnit.WORD):
        self.builder = LatticeBuilder(alignment_unit)
        self.alignment_unit = alignment_unit
    
    def calculate_wer(self, reference: str, prediction: str, 
                      lattice: Optional[Lattice] = None) -> float:
        """
        Calculate WER with optional lattice support
        
        Args:
            reference: Reference transcription
            prediction: Model prediction
            lattice: Optional lattice for valid alternatives
        
        Returns:
            WER score
        """
        if lattice is None:
            # Standard WER
            return wer([reference], [prediction])
        
        # Lattice-based WER
        return self._calculate_lattice_wer(prediction, lattice)
    
    def _calculate_lattice_wer(self, prediction: str, lattice: Lattice) -> float:
        """
        Calculate WER considering lattice alternatives
        
        The prediction is compared against the lattice where each position
        can match any of the valid alternatives.
        """
        pred_tokens = self.builder.tokenize(prediction)
        lattice_bins = lattice.to_sequence_of_bins()
        
        # Align prediction with lattice
        matches = 0
        total = max(len(pred_tokens), len(lattice_bins))
        
        for i in range(total):
            pred_token = pred_tokens[i] if i < len(pred_tokens) else ""
            valid_alts = lattice_bins[i] if i < len(lattice_bins) else []
            
            # Match if prediction matches any alternative
            if pred_token in valid_alts:
                matches += 1
        
        # WER = (S + D + I) / N
        # Simplified: (total - matches) / total
        wer_score = (total - matches) / total if total > 0 else 0
        
        return wer_score
    
    def compare_wers(self, reference: str, predictions: List[str],
                     model_outputs_for_lattice: List[str]) -> Dict[str, Dict]:
        """
        Compare traditional WER vs lattice-based WER
        
        Shows which models are unfairly penalized
        """
        results = {}
        
        # Build lattice
        lattice = self.builder.build_lattice(reference, model_outputs_for_lattice)
        
        for i, pred in enumerate(predictions):
            # Traditional WER
            trad_wer = wer([reference], [pred])
            
            # Lattice WER
            latt_wer = self._calculate_lattice_wer(pred, lattice)
            
            results[f"Model_{i}"] = {
                'prediction': pred,
                'traditional_wer': trad_wer,
                'lattice_wer': latt_wer,
                'improvement': trad_wer - latt_wer,
                'was_unfairly_penalized': (trad_wer - latt_wer) > 0.1
            }
        
        return results


def main():
    """Demonstrate lattice-based WER"""
    logger.info("Starting Question 4: Lattice-based WER Evaluation")
    
    # Example: 5 models transcribing same audio
    reference = "उसने चौदह कताबें खरीदीं"
    
    model_outputs = [
        "उसने चौदह कताबें खरीदीं",           # Correct
        "उसने 14 कताबें खरीदीं",             # Number as digit (valid)
        "उसने चौदह किताबें खरीदीं",         # Alternate spelling
        "उसने चौदा कताबें खरीदी",           # Minor errors
        "उसने शौदह कताबें खरीदीं",          # Substitution error
    ]
    
    # Build calculator
    calc = LatticeLatticeWERCalculator(alignment_unit=AlignmentUnit.WORD)
    
    # Build lattice from all models
    builder = LatticeBuilder()
    lattice = builder.build_lattice(reference, model_outputs, consensus_threshold=0.4)
    
    logger.info("\nBuilt Lattice:")
    for i, node in enumerate(lattice.nodes):
        logger.info(f"  Position {i}: {node.alternatives}")
    
    # Compare WERs
    logger.info("\nComparing Traditional vs Lattice-based WER:")
    results = calc.compare_wers(reference, model_outputs, model_outputs)
    
    for model_name, metrics in results.items():
        logger.info(f"\n{model_name}:")
        logger.info(f"  Prediction: {metrics['prediction']}")
        logger.info(f"  Traditional WER: {metrics['traditional_wer']:.3f}")
        logger.info(f"  Lattice WER: {metrics['lattice_wer']:.3f}")
        logger.info(f"  Improvement: {metrics['improvement']:.3f}")
        logger.info(f"  Unfairly penalized: {metrics['was_unfairly_penalized']}")
    
    logger.info("\nQuestion 4 pipeline complete!")


if __name__ == "__main__":
    main()
