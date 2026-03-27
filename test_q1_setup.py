"""Quick test to verify Q1 pipeline is working"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from question_1_whisper_finetuning.q1_baseline import SyntheticDatasetProcessor, ErrorAnalyzer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Testing Q1 Pipeline...")

# Test 1: Load dataset
logger.info("\n[TEST 1] Loading synthetic dataset...")
processor = SyntheticDatasetProcessor()
dataset = processor.load_dataset(max_samples=3)

if dataset and len(dataset) > 0:
    logger.info(f"✓ Dataset loaded: {len(dataset)} samples")
    for i, sample in enumerate(dataset):
        logger.info(f"  Sample {i+1}: {sample['transcription'][:50]}")
else:
    logger.error("✗ Failed to load dataset")
    sys.exit(1)

# Test 2: Error analyzer
logger.info("\n[TEST 2] Testing error analyzer...")
predictions = [
    "नमस्ते मेरा नाम राज है गलत",
    "आज का मौसम अच्छा है",
    "कंप्यूटर बहुत अच्छा है"
]
references = [
    "नमस्ते मेरा नाम राज है",
    "आज का मौसम बहुत अच्छा है",
    "मुझे कंप्यूटर प्रोग्रामिंग पसंद है"
]

errors = ErrorAnalyzer.find_errors(predictions, references)
logger.info(f"✓ Found {len(errors)} errors")

sampled = ErrorAnalyzer.sample_errors(errors, n_samples=2)
logger.info(f"✓ Sampled {len(sampled)} errors")

logger.info("\n✓ All tests passed!")
logger.info("\nNext: Run the full baseline evaluation with:")
logger.info("  python question_1_whisper_finetuning/q1_baseline.py")
