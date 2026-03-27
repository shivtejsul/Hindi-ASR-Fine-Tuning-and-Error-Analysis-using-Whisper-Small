"""
Question 2: ASR Output Cleanup Pipeline
========================================

Tasks:
a) Number Normalization: Convert Hindi number words to digits
   - Simple: दो → 2, दस → 10, सौ → 100
   - Compound: तीस → 30, तीन सौ चौवन → 354
   - Edge cases: Handle idioms (दो-चार बातें)

b) English Word Detection: Identify English words in Hindi text
   - Mark English words with tags
   - Handle transliterated English (कं प्यूटर)
"""

import re
from typing import List, Dict, Tuple, Optional
import pandas as pd
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CleanupExample:
    """Record of a cleanup operation"""
    original: str
    cleaned: str
    operation: str
    category: str
    notes: str


class HindiNumberNormalizer:
    """Convert Hindi number words to digits"""
    
    def __init__(self):
        # Hindi number words
        self.ones = {
            'शून्य': '0', 'एक': '1', 'दो': '2', 'तीन': '3', 'चार': '4',
            'पाँच': '5', 'पांच': '5', 'छह': '6', 'सात': '7', 'आठ': '8',
            'नौ': '9', 'दस': '10'
        }
        
        self.tens = {
            'बीस': '20', 'तीस': '30', 'चालीस': '40', 'पचास': '50',
            'साठ': '60', 'सत्तर': '70', 'अस्सी': '80', 'नब्बे': '90'
        }
        
        self.scales = {
            'सौ': 100,
            'सो': 100,
            'हज़ार': 1000,
            'हजार': 1000,
            'लाख': 100000,
            'करोड़': 10000000,
            'करोड': 10000000
        }
        
        # Special cases (teens) and composite numbers
        self.teens = {
            'ग्यारह': '11', 'बारह': '12', 'तेरह': '13', 'चौदह': '14',
            'पंद्रह': '15', 'सोलह': '16', 'सत्रह': '17', 'अठारह': '18',
            'उन्नीस': '19', 'पच्चीस': '25', 'छब्बीस': '26', 'सत्ताइस': '27',
            'अट्ठाइस': '28', 'उनतीस': '29', 'इकतीस': '31', 'बत्तीस': '32',
            'तेतीस': '33', 'चौंतीस': '34', 'पैंतीस': '35', 'छत्तीस': '36',
            'सैंतीस': '37', 'अड़तीस': '38', 'उनचालीस': '39', 'इकचालीस': '41',
            'बयालीस': '42', 'तेवालीस': '43', 'चौवालीस': '44', 'पैंतालीस': '45',
            'छियालीस': '46', 'सैंतालीस': '47', 'अड़तालीस': '48', 'उनचास': '49',
            'इक्यावन': '51', 'बावन': '52', 'तिरपन': '53', 'चौपन': '54',
            'पचपन': '55', 'छप्पन': '56', 'सत्तावन': '57', 'अठ्ठावन': '58',
            'उनसठ': '59', 'इकसठ': '61', 'बासठ': '62', 'तिरसठ': '63',
            'चौंसठ': '64', 'पैंसठ': '65', 'छियासठ': '66', 'सड़सठ': '67',
            'अड़सठ': '68', 'उनहत्तर': '69', 'इकहत्तर': '71', 'बहत्तर': '72',
            'तिहत्तर': '73', 'चौहत्तर': '74', 'पचहत्तर': '75', 'छिहत्तर': '76',
            'सत्तहत्तर': '77', 'अठहत्तर': '78', 'उनासी': '79', 'इक्यासी': '81',
            'बयासी': '82', 'तिरासी': '83', 'चौरासी': '84', 'पचासी': '85',
            'छियासी': '86', 'सत्तासी': '87', 'अठ्ठासी': '88', 'उनानवे': '89',
            'इक्यानवे': '91', 'बानवे': '92', 'तिरानवे': '93', 'चौरानवे': '94',
            'पचानवे': '95', 'छियानवे': '96', 'सत्तानवे': '97', 'अठ्ठानवे': '98',
            'निन्यानवे': '99', 'चौवन': '54'
        }
    
    def is_number_context(self, text: str, pos: int) -> bool:
        """
        Check if position is in a number context or an idiom
        
        Examples of idioms to preserve:
        - दो-चार बातें (2-4 matters) → keep as "दो-चार"
        - एक-दो दिन (one-two days) → keep as "एक-दो"
        """
        # Check if part of a hyphenated idiom pattern
        before = text[:pos].rstrip()
        after = text[pos:].lstrip()
        
        # Pattern: number-number + noun (idiom pattern)
        if re.search(r'[\w-]+$', before) and re.search(r'^[\w-]+', after):
            # Could be an idiom, analyze more
            nearby = text[max(0, pos-10):min(len(text), pos+10)]
            if '-' in nearby:
                return False  # Likely an idiom, don't convert
        
        return True
    
    def normalize_compound_numbers(self, text: str) -> str:
        """
        Convert compound numbers like:
        - तीन सौ चौवन → 354
        - एक हज़ार → 1000
        - पचास → 50
        """
        result = text
        
        # Create a mapping of all number words to values
        all_numbers = {}
        for word, digit in self.ones.items():
            all_numbers[word] = int(digit) if digit.isdigit() else digit
        for word, digit in self.tens.items():
            all_numbers[word] = int(digit)
        for word, digit in self.teens.items():
            all_numbers[word] = int(digit)
        for word, value in self.scales.items():
            all_numbers[word] = value
        
        # Split into words to parse compound numbers
        words = result.split()
        output_words = []
        i = 0
        
        while i < len(words):
            word = words[i]
            
            # Check if this word starts a compound number sequence
            # Try to match: number number number ... scale
            number_sequence = []
            j = i
            
            while j < len(words):
                # Remove punctuation for checking
                clean_word = re.sub(r'[।,।।!?।॥-]', '', words[j])
                
                if clean_word in all_numbers:
                    number_sequence.append((clean_word, all_numbers[clean_word]))
                    j += 1
                elif j > i:  # We have at least one number word
                    break
                else:
                    break
            
            # If we found a number sequence
            if len(number_sequence) > 0:
                # Check if it's part of an idiom (hyphenated pattern)
                is_idiom = False
                if i > 0 and '-' in words[i-1]:
                    is_idiom = True  # Previous word ends with hyphen
                if i + len(number_sequence) < len(words) and '-' in words[i + len(number_sequence)]:
                    is_idiom = True  # Next word starts with hyphen
                
                if is_idiom:
                    # Don't convert idioms like "दो-चार"
                    output_words.append(word)
                    i += 1
                else:
                    # Convert compound number to digit
                    total = self._parse_compound_number(number_sequence)
                    output_words.append(str(total))
                    i = j
            else:
                output_words.append(word)
                i += 1
        
        return ' '.join(output_words)
    
    def _parse_compound_number(self, number_sequence: List[Tuple[str, int]]) -> int:
        """
        Parse a sequence of number words into a single value
        
        Examples:
        - [(तीन, 3), (सौ, 100), (चौवन, 54)] → 354
        - [(एक, 1), (हज़ार, 1000)] → 1000
        """
        if not number_sequence:
            return 0
        
        total = 0
        current = 0
        
        for word, value in number_sequence:
            if value >= 100:  # Scale word
                current = (current or 1) * value
                total += current
                current = 0
            else:  # Regular number
                current += value
        
        total += current
        return total if total > 0 else int(number_sequence[0][1])
    
    def normalize_simple_numbers(self, text: str) -> str:
        """Convert simple number words, preserving idioms"""
        words = text.split()
        output_words = []
        
        for i, word in enumerate(words):
            # Check if this is part of a hyphenated idiom (दो-चार pattern)
            is_idiom = False
            if '-' in word:
                # Word contains hyphen - likely an idiom, don't normalize
                is_idiom = True
            elif i > 0 and words[i-1].endswith('-'):
                # Previous word ends with hyphen
                is_idiom = True
            elif i < len(words) - 1 and words[i+1].startswith('-'):
                # Next word starts with hyphen
                is_idiom = True
            
            if is_idiom:
                # Don't convert idioms
                output_words.append(word)
            else:
                # Try to convert simple number words
                clean_word = re.sub(r'[।,।।!?।॥]', '', word)
                converted = False
                
                for num_word, digit in {**self.ones, **self.tens, **self.teens}.items():
                    if clean_word == num_word or word == num_word:
                        # Replace preserving any punctuation
                        output_words.append(digit)
                        converted = True
                        break
                
                if not converted:
                    output_words.append(word)
        
        return ' '.join(output_words)
    
    def normalize_numbers(self, text: str, handle_idioms: bool = True) -> str:
        """Main normalization function"""
        # Step 1: Handle compound numbers first (before simple numbers)
        text = self.normalize_compound_numbers(text)
        
        # Step 2: Handle remaining simple numbers
        text = self.normalize_simple_numbers(text)
        
        return text


class EnglishWordDetector:
    """Detect and tag English words in Hindi text"""
    
    def __init__(self):
        # Common English words that appear in Hindi conversations
        self.english_words = {
            'interview', 'problem', 'job', 'computer', 'software',
            'email', 'mobile', 'phone', 'ok', 'yes', 'no', 'hello',
            'name', 'address', 'company', 'project', 'meeting',
            'boss', 'manager', 'team', 'work', 'office', 'time'
        }
        
        # Common transliterations (Devanagari)
        self.transliterations = {
            'इंटरव्यू': 'interview',
            'प्रॉब्लम': 'problem',
            'कंप्यूटर': 'computer',
            'कम्प्यूटर': 'computer',
            'सॉफ्टवेयर': 'software',
            'इमेल': 'email',
            'मोबाइल': 'mobile',
            'फोन': 'phone',
            'जॉब': 'job',
            'मीटिंग': 'meeting',
            'मैनेजर': 'manager',
        }
    
    def detect_english_words(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Detect English words in Hindi text
        
        Returns:
            List of (start_pos, end_pos, word) tuples
        """
        english_positions = []
        
        # Split text into words
        words = text.split()
        current_pos = 0
        
        for word in words:
            clean_word = re.sub(r'[।,።!?॥]', '', word).lower()
            
            # Check if it's an English word (mostly Latin characters)
            if re.search(r'^[a-zA-Z]+$', clean_word):
                start = text.find(word, current_pos)
                end = start + len(word)
                english_positions.append((start, end, word))
            
            # Check if it's a known transliteration
            elif clean_word in self.transliterations or word in self.transliterations:
                start = text.find(word, current_pos)
                end = start + len(word)
                english_positions.append((start, end, word))
            
            current_pos += len(word) + 1
        
        return english_positions
    
    def tag_english_words(self, text: str, tag_format: str = "xml") -> str:
        """
        Tag English words in the text
        
        Args:
            text: Input Hindi text with English words
            tag_format: "xml" for [EN]word[/EN] or "other" formats
        
        Returns:
            Tagged text
        """
        english_positions = self.detect_english_words(text)
        
        if not english_positions:
            return text
        
        # Sort by position (reverse order to maintain indices)
        english_positions = sorted(english_positions, key=lambda x: x[0], reverse=True)
        
        result = text
        for start, end, word in english_positions:
            if tag_format == "xml":
                tagged = f"[EN]{result[start:end]}[/EN]"
                result = result[:start] + tagged + result[end:]
            elif tag_format == "parentheses":
                tagged = f"({result[start:end]})"
                result = result[:start] + tagged + result[end:]
        
        return result


class AsrCleanupPipeline:
    """Main cleanup pipeline combining normalization and detection"""
    
    def __init__(self):
        self.normalizer = HindiNumberNormalizer()
        self.detector = EnglishWordDetector()
        self.examples = []
    
    def cleanup(self, text: str, normalize_numbers: bool = True, 
                tag_english: bool = True) -> str:
        """
        Full cleanup pipeline
        """
        original = text
        
        # Step 1: Normalize numbers
        if normalize_numbers:
            text = self.normalizer.normalize_numbers(text)
        
        # Step 2: Tag English words
        if tag_english:
            text = self.detector.tag_english_words(text)
        
        # Log example
        if original != text:
            self.examples.append(CleanupExample(
                original=original,
                cleaned=text,
                operation="normalize_and_tag",
                category="mixed",
                notes=""
            ))
        
        return text
    
    def process_batch(self, texts: List[str]) -> List[str]:
        """Process a batch of texts"""
        return [self.cleanup(text) for text in texts]
    
    def evaluate_cleanup(self, texts: List[str], references: List[str]) -> Dict:
        """
        Evaluate cleanup effectiveness:
        - Does it help or hurt?
        - Track metrics for different operations
        """
        results = {
            'total_texts': len(texts),
            'texts_changed': 0,
            'examples_before_after': []
        }
        
        for text, ref in zip(texts, references):
            cleaned = self.cleanup(text)
            
            if cleaned != text:
                results['texts_changed'] += 1
                results['examples_before_after'].append({
                    'before': text,
                    'after': cleaned,
                    'reference': ref
                })
        
        return results


def main():
    """Test the cleanup pipeline"""
    logger.info("Starting Question 2: ASR Cleanup Pipeline")
    
    # Example texts to clean
    example_texts = [
        "मेरा इंटरव्यू कल है और मुझे जॉब मल गई",
        "उसने तीन सौ चौवन कताबें खरीदीं",
        "यह प्रॉब्लम सॉल्व नहीं हो रहा है",
        "मेरे पास दो-चार बातें कहने के लिए हैं",
        "एक हज़ार रुपये खर्च हुए",
    ]
    
    pipeline = AsrCleanupPipeline()
    
    logger.info("\nTesting Number Normalization:")
    normalizer = HindiNumberNormalizer()
    for text in example_texts:
        normalized = normalizer.normalize_numbers(text)
        logger.info(f"  Before: {text}")
        logger.info(f"  After:  {normalized}\n")
    
    logger.info("\nTesting English Word Detection:")
    detector = EnglishWordDetector()
    for text in example_texts:
        tagged = detector.tag_english_words(text)
        logger.info(f"  Before: {text}")
        logger.info(f"  After:  {tagged}\n")
    
    logger.info("\nTesting Full Pipeline:")
    for text in example_texts:
        cleaned = pipeline.cleanup(text)
        logger.info(f"  Before: {text}")
        logger.info(f"  After:  {cleaned}\n")


if __name__ == "__main__":
    main()
