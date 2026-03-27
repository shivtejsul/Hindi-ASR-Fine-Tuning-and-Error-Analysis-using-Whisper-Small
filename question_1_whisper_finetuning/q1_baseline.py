"""
Question 1: Whisper Fine-tuning & Error Analysis for Hindi ASR
Working with Synthetic Dataset

Tasks:
a) Preprocess the dataset
b) Fine-tune Whisper-tiny
c) Evaluate on FLEURS Hindi test set
d) Sample 25+ error utterances systematically
e) Build error taxonomy with concrete examples
f) Propose fixes for top 3 error types
g) Implement at least one fix with before/after results
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Dataset
from jiwer import wer, cer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticDatasetProcessor:
    """Load synthetic dataset from local files"""
    
    def __init__(self, metadata_path: str = "data/synthetic/metadata.csv"):
        self.metadata_path = metadata_path
        self.df = None
    
    def load_metadata(self) -> bool:
        """Load metadata CSV"""
        try:
            self.df = pd.read_csv(self.metadata_path)
            logger.info(f"Loaded {len(self.df)} records from metadata")
            return True
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return False
    
    def preprocess_audio(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Preprocess audio"""
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # Resample if needed
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        return audio
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess transcription text"""
        text = text.strip()
        return text
    
    def load_dataset(self, max_samples: Optional[int] = None) -> Optional[Dataset]:
        """Load dataset from local files"""
        if self.df is None:
            if not self.load_metadata():
                return None
        
        data = []
        df_subset = self.df.iloc[:max_samples] if max_samples else self.df
        
        for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="Loading samples"):
            try:
                # Load audio
                audio, sr = librosa.load(row['audio_path'], sr=16000)
                audio = self.preprocess_audio(audio)
                
                # Get transcription
                text = self.preprocess_text(row['transcription'])
                
                if len(text) > 0 and len(audio) > 0:
                    data.append({
                        'user_id': row['user_id'],
                        'recording_id': row['recording_id'],
                        'audio': {'array': audio, 'sampling_rate': 16000},
                        'transcription': text,
                        'duration': len(audio) / 16000
                    })
            except Exception as e:
                logger.warning(f"Error loading sample {idx}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(data)} samples")
        return Dataset.from_list(data) if data else None


class WhisperBaseline:
    """Load pretrained Whisper for baseline evaluation"""
    
    def __init__(self, model_name: str = "openai/whisper-tiny", device: str = "cpu"):
        self.device = device
        self.model_name = model_name
        
        logger.info(f"Loading {model_name} on device: {device}...")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        # Force Hindi language for transcription
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="hi", task="transcribe")
        logger.info("✓ Model loaded successfully (Hindi forced)")
    
    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio"""
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(inputs.input_features)
        
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription
    
    def evaluate(self, test_dataset: Dataset) -> Dict[str, float]:
        """Evaluate model on test set"""
        predictions = []
        references = []
        
        for sample in tqdm(test_dataset, desc="Evaluating"):
            try:
                pred = self.transcribe(sample['audio']['array'])
                predictions.append(pred)
                references.append(sample['transcription'])
            except Exception as e:
                logger.warning(f"Error evaluating sample: {e}")
                continue
        
        if len(predictions) == 0:
            logger.error("No predictions made")
            return {}
        
        # Calculate metrics
        wer_score = wer(references, predictions)
        cer_score = cer(references, predictions)
        
        return {
            "wer": wer_score,
            "cer": cer_score,
            "predictions": predictions,
            "references": references,
            "num_samples": len(predictions)
        }


class ErrorAnalyzer:
    """Analyze transcription errors"""
    
    @staticmethod
    def find_errors(predictions: List[str], references: List[str]) -> List[Dict]:
        """Find and characterize errors"""
        errors = []
        
        for pred, ref in zip(predictions, references):
            if pred.lower().strip() != ref.lower().strip():
                errors.append({
                    'prediction': pred,
                    'reference': ref,
                    'cer': cer([ref], [pred]),
                    'wer': wer([ref], [pred])
                })
        
        return errors
    
    @staticmethod
    def sample_errors(errors: List[Dict], n_samples: int = 25, 
                      strategy: str = "random") -> List[Dict]:
        """Sample errors systematically"""
        if not errors:
            return []
        
        n = min(n_samples, len(errors))
        
        if strategy == "random":
            import random
            sampled = random.sample(errors, n)
        elif strategy == "stratified":
            # Sort by severity and stratify
            sorted_errors = sorted(errors, key=lambda x: x['cer'])
            step = max(len(sorted_errors) // n, 1)
            sampled = sorted_errors[::step][:n]
        else:
            sampled = errors[:n]
        
        return sampled
    
    @staticmethod
    def build_error_report(error_samples: List[Dict]) -> str:
        """Build error analysis report"""
        report = "\n" + "="*70 + "\n"
        report += "ERROR ANALYSIS REPORT\n"
        report += "="*70 + "\n\n"
        report += f"Total errors analyzed: {len(error_samples)}\n"
        report += f"Average WER: {np.mean([e['wer'] for e in error_samples]):.3f}\n"
        report += f"Average CER: {np.mean([e['cer'] for e in error_samples]):.3f}\n\n"
        
        report += "SAMPLE ERRORS:\n"
        report += "-"*70 + "\n"
        
        for i, error in enumerate(error_samples[:10], 1):
            report += f"\nError {i}:\n"
            report += f"  Reference: {error['reference']}\n"
            report += f"  Predicted: {error['prediction']}\n"
            report += f"  WER: {error['wer']:.3f}, CER: {error['cer']:.3f}\n"
        
        report += "\n" + "="*70 + "\n"
        return report


def main():
    """Main execution pipeline for Question 1"""
    logger.info("="*70)
    logger.info("Question 1: Whisper Fine-tuning & Error Analysis (Synthetic Data)")
    logger.info("="*70)
    
    # Step 1: Load dataset
    logger.info("\n[STEP 1] Loading synthetic dataset...")
    processor = SyntheticDatasetProcessor()
    dataset = processor.load_dataset()
    
    if dataset is None or len(dataset) == 0:
        logger.error("Failed to load dataset. Exiting.")
        return
    
    logger.info(f"✓ Loaded {len(dataset)} samples")
    logger.info(f"  Sample 1: '{dataset[0]['transcription']}'")
    logger.info(f"  Duration: {dataset[0]['duration']:.1f}s")
    
    # Step 2: Load pretrained Whisper
    logger.info("\n[STEP 2] Loading pretrained Whisper-tiny...")
    try:
        whisper = WhisperBaseline()
    except Exception as e:
        logger.error(f"Failed to load Whisper: {e}")
        return
    
    # Step 3: Evaluate baseline on full dataset
    logger.info("\n[STEP 3] Evaluating baseline Whisper...")
    results = whisper.evaluate(dataset)
    
    if not results:
        logger.error("Evaluation failed")
        return
    
    logger.info(f"✓ Evaluation complete!")
    logger.info(f"  Word Error Rate (WER): {results['wer']:.4f}")
    logger.info(f"  Character Error Rate (CER): {results['cer']:.4f}")
    logger.info(f"  Samples evaluated: {results['num_samples']}")
    
    # Step 4: Analyze errors
    logger.info("\n[STEP 4] Analyzing errors...")
    analyzer = ErrorAnalyzer()
    errors = analyzer.find_errors(results['predictions'], results['references'])
    
    logger.info(f"✓ Found {len(errors)} errors out of {results['num_samples']} samples")
    
    if errors:
        # Sample errors
        sampled_errors = analyzer.sample_errors(errors, n_samples=min(25, len(errors)))
        report = analyzer.build_error_report(sampled_errors)
        logger.info(report)
        
        # Save report
        report_path = "results/q1_error_analysis.txt"
        os.makedirs("results", exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"\n✓ Error report saved to {report_path}")
    
    # Step 5: Summary and next steps
    logger.info("\n" + "="*70)
    logger.info("BASELINE EVALUATION COMPLETE")
    logger.info("="*70)
    logger.info(f"""
Summary:
--------
✓ Loaded {len(dataset)} Hindi ASR samples
✓ Baseline WER: {results['wer']:.4f}
✓ Found {len(errors)} transcription errors
✓ Error report saved to results/q1_error_analysis.txt

Next Steps:
-----------
1. Fine-tune Whisper on Hindi training data
2. Evaluate fine-tuned model
3. Compare with baseline
4. Build comprehensive error taxonomy
5. Implement error fixes

To continue with fine-tuning, update this script or run:
  python question_1_whisper_finetuning/q1_finetune.py
    """)


if __name__ == "__main__":
    main()
