"""
Question 3: Spelling Error Classification
===========================================

Tasks:
a) Classify ~177,000 unique words as correct/incorrect spelling
b) Output confidence scores (high/medium/low)
c) Review 40-50 low-confidence words and analyze accuracy
d) Identify word categories where system is unreliable

Deliverables:
- Google Sheet with word list and classification
- Count of correct spelled words
- Analysis of broken cases
"""

import json
from typing import List, Dict, Tuple, Optional
import pandas as pd
from dataclasses import dataclass
import logging
import unicodedata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WordClassification:
    """Classification of a single word"""
    word: str
    classification: str  # "correct" or "incorrect"
    confidence: str     # "high", "medium", "low"
    reason: str
    category: Optional[str] = None


class HindiSpellingClassifier:
    """Classify Hindi words as correctly or incorrectly spelled"""
    
    def __init__(self, dictionary_path: Optional[str] = None):
        """
        Initialize classifier
        
        Args:
            dictionary_path: Path to Hindi dictionary (if available)
        """
        self.classifications = []
        
        # Built-in common Hindi words (basic dictionary for testing)
        self.built_in_hindi = {
            'आज', 'कल', 'आना', 'जाना', 'देना', 'लेना', 'करना', 'होना',
            'खरीदी', 'खरीदीं', 'कताबें', 'किताबें', 'बातें', 'दिन', 'समय',
            'शिक्षा', 'विद्यार्थी', 'शिक्षक', 'स्कूल', 'कॉलेज', 'विश्वविद्यालय',
            'किताब', 'पुस्तक', 'लेखक', 'रचना', 'काव्य', 'नाटक', 'कविता',
            'भारत', 'दिल्ली', 'मुंबई', 'बेंगलुरु', 'हैदराबाद', 'चेन्नई',
            'अच्छा', 'बुरा', 'बड़ा', 'छोटा', 'लंबा', 'छोटी', 'बड़ी',
            'माता', 'पिता', 'भाई', 'बहन', 'दादा', 'दादी', 'नाना', 'नानी',
            'घर', 'दरवाज़ा', 'खिड़की', 'छत', 'दीवार', 'फर्श', 'छत',
            'खाना', 'पानी', 'चाय', 'दूध', 'रोटी', 'चावल', 'दाल',
            'पहनना', 'उतारना', 'धोना', 'सूखना', 'रंग', 'कपड़े', 'जूते',
            'मौसम', 'गर्मी', 'सर्दी', 'बारिश', 'हवा', 'बादल', 'सूरज',
            'रात', 'दिन', 'सुबह', 'शाम', 'रात', 'आधी रात',
            'हाथ', 'पैर', 'सिर', 'आंख', 'कान', 'नाक', 'मुंह', 'दांत',
            'बीमारी', 'दर्द', 'ठीक', 'बीमार', 'स्वस्थ', 'दवा',
            'हंसना', 'रोना', 'हंसी', 'खुशी', 'दुःख', 'गुस्सा', 'प्रेम',
            'क्षेत्र', 'ज्ञान', 'बुद्धि', 'विचार', 'विचार', 'तर्क',
            'महत्वपूर्ण', 'आवश्यक', 'उपयोगी', 'मूल्यवान', 'कीमती',
        }
        
        self.dictionary = self._load_dictionary(dictionary_path) if dictionary_path else self.built_in_hindi
        
        # Known patterns for common spelling mistakes in Hindi
        self.error_patterns = {
            'duplicate_vowels': r'([ािीुूेैोौ])\1{2,}',  # Triple vowel marks
            'missing_nukta': r'[कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह](?=[ा])',
            'incorrect_ra_kaar': r'([^\u093C])(ृ|ॄ)',  # Wrong ra-kaar usage
        }
        
        # Common typos and their corrections
        self.known_corrections = {
            'महत्वपूर्ण': 'महत्वपूर्ण',  # Common word
            'तकनीकी': 'तकनीकी',
        }
    
    def _load_dictionary(self, path: str) -> set:
        """Load Hindi dictionary from file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                words = set(line.strip() for line in f)
            logger.info(f"Loaded {len(words)} words from dictionary")
            return words
        except Exception as e:
            logger.warning(f"Failed to load dictionary: {e}")
            return set()
    
    def is_valid_devanagari(self, word: str) -> bool:
        """Check if word contains valid Devanagari characters"""
        pattern = r'^[\u0900-\u097F\-]+$'  # Devanagari range + hyphen
        return bool(re.match(pattern, word))
    
    def is_english_transliteration(self, word: str) -> bool:
        """
        Check if word is an English word transliterated to Devanagari
        These should NOT be marked as spelling errors
        
        Examples:
        - कंप्यूटर (computer)
        - इंटरव्यू (interview)
        - प्रोग्राम (program)
        
        Strategy: Check known transliterations and specific patterns
        """
        # Known English transliterations (tech/modern terms)
        known_english_transliterations = {
            'इंटरव्यू', 'कंप्यूटर', 'कम्प्यूटर', 'सॉफ्टवेयर',
            'इमेल', 'मोबाइल', 'फोन', 'जॉब', 'मीटिंग',
            'प्रोग्राम', 'डेवेलपर', 'सॉल्व', 'डाटा', 'डेटा',
            'नेटवर्क', 'सर्वर', 'क्लाइंट', 'एप्लिकेशन', 'टेस्ट'
        }
        
        if word in known_english_transliterations:
            return True
        
        # Check for specific transliteration patterns:
        # 1. Multiple consonant clusters (TV/consonant + halant combinations)
        # 2. Rare in pure Hindi but common in English words
        
        # Pattern: consonant्consonant with multiple occurrences = likely English
        halant_clusters = len(re.findall(r'्', word))
        
        # In English transliterations, we often see:
        # - Multiple halants (consonant clusters)
        # - Words ending with consonant clusters
        # - Presence of specific consonants like ड़, ख़, ज़ (used in foreign words)
        
        foreign_consonants = {'ड़', 'ख़', 'ज़', 'फ़', 'य़'}
        has_foreign = any(c in word for c in foreign_consonants)
        
        # A word with both foreign consonants AND multiple consonant clusters is likely English
        if has_foreign and halant_clusters >= 1:
            return True
        
        # Otherwise, assume it's a regular Hindi word
        return False
    
    def classify_word(self, word: str) -> WordClassification:
        """
        Classify a single word as correct or incorrect
        
        Returns:
            WordClassification with confidence score
        """
        word = word.strip()
        
        # Check basic validity
        if not word:
            return WordClassification(
                word=word, classification="incorrect",
                confidence="high", reason="Empty word"
            )
        
        if not self.is_valid_devanagari(word):
            return WordClassification(
                word=word, classification="incorrect",
                confidence="high", reason="Invalid Devanagari characters"
            )
        
        # Check if it's an English transliteration
        if self.is_english_transliteration(word):
            return WordClassification(
                word=word, classification="correct",
                confidence="high", reason="Valid English transliteration",
                category="transliteration"
            )
        
        # Check against dictionary
        if word in self.dictionary:
            return WordClassification(
                word=word, classification="correct",
                confidence="high", reason="Found in dictionary",
                category="dictionary_word"
            )
        
        # Check for known corrections
        if word in self.known_corrections:
            return WordClassification(
                word=word, classification="correct",
                confidence="high", reason="Known correct word",
                category="known_word"
            )
        
        # Heuristic analysis
        if self._has_suspicious_patterns(word):
            return WordClassification(
                word=word, classification="incorrect",
                confidence="medium", 
                reason="Suspicious character patterns detected",
                category="pattern_error"
            )
        
        # If unknown, default to low confidence
        return WordClassification(
            word=word, classification="unknown",
            confidence="low",
            reason="Not found in dictionary, needs manual review",
            category="unknown"
        )
    
    def _has_suspicious_patterns(self, word: str) -> bool:
        """Check for suspicious patterns indicating spelling errors"""
        import re
        
        # Check for triple+ vowel marks (rare in correct Hindi)
        if re.search(r'([ािीुूेैोौ])\1{2,}', word):
            return True
        
        # Check for unusual consonant clusters
        consonant_cluster_pattern = r'[क-ह][्][क-ह][्][क-ह]'
        if re.search(consonant_cluster_pattern, word):
            return True
        
        return False
    
    def classify_batch(self, words: List[str]) -> List[WordClassification]:
        """Classify a batch of words"""
        classifications = []
        for word in words:
            classifications.append(self.classify_word(word))
        return classifications
    
    def analyze_low_confidence(self, classifications: List[WordClassification], 
                               sample_size: int = 50) -> Dict:
        """
        Sample and analyze low-confidence classifications
        
        Return accuracy metrics and error analysis
        """
        low_conf = [c for c in classifications if c.confidence == "low"]
        
        logger.info(f"Total low confidence items: {len(low_conf)}")
        
        # Sample
        import random
        sample = random.sample(low_conf, min(sample_size, len(low_conf)))
        
        analysis = {
            'total_low_confidence': len(low_conf),
            'sample_size': len(sample),
            'sample': sample,
            'manual_review_needed': True,  # Would need human review for accuracy
        }
        
        return analysis


class SpellingClassificationPipeline:
    """Main pipeline for word classification"""
    
    def __init__(self, word_list_path: Optional[str] = None):
        self.classifier = HindiSpellingClassifier()
        self.words = self._load_word_list(word_list_path) if word_list_path else []
        self.classifications = []
    
    def _load_word_list(self, path: str) -> List[str]:
        """Load word list from file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                words = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(words)} unique words")
            return words
        except Exception as e:
            logger.error(f"Failed to load word list: {e}")
            return []
    
    def classify_all_words(self, words: Optional[List[str]] = None) -> List[WordClassification]:
        """Classify all words"""
        if words:
            self.words = words
        
        logger.info(f"Classifying {len(self.words)} words...")
        
        self.classifications = []
        for word in self.words:
            classification = self.classifier.classify_word(word)
            self.classifications.append(classification)
        
        return self.classifications
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Export classifications to DataFrame for Google Sheets"""
        data = []
        
        for c in self.classifications:
            data.append({
                'Word': c.word,
                'Classification': c.classification,
                'Confidence': c.confidence,
                'Reason': c.reason,
                'Category': c.category or ''
            })
        
        return pd.DataFrame(data)
    
    def export_to_csv(self, output_path: str):
        """Export to CSV (can be imported to Google Sheets)"""
        df = self.export_to_dataframe()
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"Exported {len(df)} words to {output_path}")
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics"""
        stats = {
            'total_words': len(self.classifications),
            'correct_spelled': sum(1 for c in self.classifications if c.classification == "correct"),
            'incorrect_spelled': sum(1 for c in self.classifications if c.classification == "incorrect"),
            'unknown': sum(1 for c in self.classifications if c.classification == "unknown"),
            'high_confidence': sum(1 for c in self.classifications if c.confidence == "high"),
            'medium_confidence': sum(1 for c in self.classifications if c.confidence == "medium"),
            'low_confidence': sum(1 for c in self.classifications if c.confidence == "low"),
        }
        
        stats['correct_percentage'] = (stats['correct_spelled'] / stats['total_words'] * 100) if stats['total_words'] > 0 else 0
        
        return stats
    
    def analyze_unreliable_categories(self) -> Dict[str, Dict]:
        """Identify word categories where system is unreliable"""
        unreliable = {}
        
        # Group by category
        by_category = {}
        for c in self.classifications:
            cat = c.category or "uncategorized"
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(c)
        
        # Analyze each category
        for category, items in by_category.items():
            total = len(items)
            low_conf = sum(1 for i in items if i.confidence == "low")
            
            unreliable[category] = {
                'total_items': total,
                'low_confidence_items': low_conf,
                'low_confidence_percentage': low_conf / total * 100 if total > 0 else 0,
                'reason': self._get_category_reason(category)
            }
        
        return unreliable
    
    def _get_category_reason(self, category: str) -> str:
        """Provide reasoning for unreliability"""
        reasons = {
            'transliteration': 'Variable transliteration conventions, script normalization needed',
            'unknown': 'Words not in dictionary - could be neologisms, regional variants, or typos',
            'pattern_error': 'Suspicious patterns detected but needs contextual understanding',
            'proper_noun': 'Named entities, hard to validate without named entity database',
        }
        return reasons.get(category, "Unknown reason")


def main():
    """Main execution"""
    logger.info("Starting Question 3: Spelling Error Classification")
    
    # Initialize pipeline
    pipeline = SpellingClassificationPipeline()
    
    # For testing, use sample words demonstrating different categories
    sample_words = [
        # Correct Hindi words
        'महत्वपूर्ण', 'आज', 'कल', 'शिक्षा', 'विद्यार्थी', 'घर', 'खाना',
        # English transliterations (correct in Hindi)
        'कंप्यूटर', 'इंटरव्यू', 'सॉफ्टवेयर', 'प्रोग्राम', 'डेवेलपर',
        # Misspelled words
        'विदयार्थी',  # Missing one diacritic
        'शिक्षा',  # Intentional misspell
        # Unknown words (not in dictionary)
        'नेटवर्क', 'डाटा', 'एल्गोरिदम',
        # Edge cases
        'दो-चार',  # Hyphenated idiom
        'क्षेत्र', 'ज्ञान',  # Complex consonant clusters
    ]
    
    classifications = pipeline.classify_all_words(pipeline.words if pipeline.words else sample_words)
    
    logger.info("\nClassifications (showing first 15):")
    for i, c in enumerate(classifications[:15]):
        logger.info(f"  {c.word}: {c.classification} ({c.confidence}) - {c.reason}")
    
    logger.info("\nSummary Statistics:")
    stats = pipeline.get_summary_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Analyze unreliable categories
    logger.info("\nUnreliable Categories:")
    unreliable = pipeline.analyze_unreliable_categories()
    for cat, info in unreliable.items():
        if info['low_confidence_percentage'] > 0:
            logger.info(f"  {cat}: {info['low_confidence_percentage']:.1f}% low confidence")
    
    # Export
    logger.info("\nExporting to CSV...")
    pipeline.export_to_csv("q3_word_classifications.csv")
    
    logger.info("\nQuestion 3 pipeline complete!")


if __name__ == "__main__":
    # Add missing import
    import re
    main()
