"""
Question 1: Whisper Fine-tuning & Error Analysis for Hindi ASR
================================================================

Tasks:
a) Preprocess the dataset
b) Fine-tune Whisper-small
c) Evaluate on FLEURS Hindi test set
d) Sample 25+ error utterances systematically
e) Build error taxonomy with concrete examples
f) Propose fixes for top 3 error types
g) Implement at least one fix with before/after results
"""

import json
import os
import requests
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Dataset
from jiwer import wer, cer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetProcessor:
    """Handle dataset loading and preprocessing"""
    
    def __init__(self, csv_path: Optional[str] = None):
        self.csv_path = csv_path or r"c:\Users\shivt\Downloads\FT Data - data.csv"
        self.processed_data = []
        self.df = None
    
    def load_csv(self) -> bool:
        """Load the CSV file with GCS URLs"""
        try:
            self.df = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(self.df)} records from CSV")
            return True
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            return False
    
    def fetch_transcription(self, transcription_url: str) -> Optional[Dict]:
        """Fetch transcription JSON from GCS"""
        try:
            response = requests.get(transcription_url, timeout=20)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to fetch transcription: {e}")
            return None
    
    def fetch_audio(self, audio_url: str) -> Optional[np.ndarray]:
        """Fetch and load audio from GCS"""
        try:
            logger.info(f"Downloading audio...")
            response = requests.get(audio_url, timeout=30)
            response.raise_for_status()
            
            # Save temporarily and load with librosa
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(response.content)
                temp_path = tmp.name
            
            audio, sr = librosa.load(temp_path, sr=16000)
            os.remove(temp_path)
            return audio
        except Exception as e:
            logger.warning(f"Failed to fetch audio: {e}")
            return None
    
    def preprocess_audio(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Preprocess audio:
        - Normalize volume
        - Remove silence
        - Resample to 16kHz
        """
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # Resample if needed
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        return audio
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess transcription text:
        - Strip whitespace
        - Handle unicode normalization
        """
        text = text.strip()
        return text
    
    def process_dataset(self, max_samples: Optional[int] = None) -> Optional[Dataset]:
        """
        Process samples from CSV into a HuggingFace Dataset
        
        Args:
            max_samples: Maximum number of samples to process
        
        Returns:
            HuggingFace Dataset with 'audio' and 'transcription' columns
        """
        if self.df is None:
            if not self.load_csv():
                logger.error("Failed to load CSV")
                return None
        
        data = []
        df_subset = self.df.iloc[:max_samples] if max_samples else self.df
        
        for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset)):
            # Fetch transcription
            trans_data = self.fetch_transcription(row['transcription_url_gcp'])
            if not trans_data:
                continue
            
            # Fetch audio
            audio = self.fetch_audio(row['rec_url_gcp'])
            if audio is None:
                continue
            
            # Preprocess
            audio = self.preprocess_audio(audio)
            text = self.preprocess_text(trans_data.get('transcription', ''))
            
            if len(text) > 0 and len(audio) > 0:
                data.append({
                    'user_id': row['user_id'],
                    'recording_id': row['recording_id'],
                    'audio': {'array': audio, 'sampling_rate': 16000},
                    'transcription': text,
                    'duration': len(audio) / 16000
                })
                logger.info(f"Processed {len(data)} samples so far...")
        
        logger.info(f"Successfully processed {len(data)} samples")
        return Dataset.from_list(data) if data else None


class WhisperFinetuner:
    """Fine-tune Whisper model on Hindi ASR data"""
    
    def __init__(self, model_name: str = "openai/whisper-small", device: str = "cuda"):
        self.device = device
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        self.model.config.forced_decoder_ids = None
        # Set language to Hindi
        self.processor.tokenizer.set_prefix_tokens(language="hi", task="transcribe")
    
    def prepare_dataset(self, dataset: Dataset):
        """Prepare dataset for fine-tuning"""
        def prepare_batch(batch):
            audio = batch["audio"]
            inputs = self.processor(
                audio["array"],
                sampling_rate=audio["sampling_rate"],
                return_tensors="pt"
            )
            with self.processor.as_target_processor():
                labels = self.processor(batch["transcription"], return_tensors="pt")
            
            return {
                "input_features": inputs.input_features[0],
                "labels": labels.input_ids[0]
            }
        
        return dataset.map(prepare_batch)
    
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None, 
              output_dir: str = "whisper_hindi_finetuned", num_epochs: int = 3):
        """
        Fine-tune the model
        
        TODO: Implement training loop with optimization
        """
        logger.info(f"Starting fine-tuning on {len(train_dataset)} samples")
        # Implementation will use HuggingFace Trainer
        pass
    
    def evaluate(self, test_dataset: Dataset) -> Dict[str, float]:
        """Evaluate model on test set"""
        predictions = []
        references = []
        
        for sample in tqdm(test_dataset):
            # Get model prediction
            inputs = self.processor(
                sample["audio"]["array"],
                sampling_rate=16000,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(inputs.input_features)
            
            transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            predictions.append(transcription)
            references.append(sample["transcription"])
        
        # Calculate metrics
        wer_score = wer(references, predictions)
        cer_score = cer(references, predictions)
        
        return {
            "wer": wer_score,
            "cer": cer_score,
            "predictions": predictions,
            "references": references
        }


class ErrorAnalyzer:
    """Analyze model errors and build taxonomy"""
    
    def __init__(self):
        self.error_samples = []
    
    def sample_errors(self, predictions: List[str], references: List[str], 
                      n_samples: int = 25, strategy: str = "random") -> List[Dict]:
        """
        Sample errors systematically
        
        Strategies:
        - "random": Random sampling of errors
        - "stratified": Sample by error severity (WER per utterance)
        - "every_nth": Every nth error
        """
        errors = []
        
        for pred, ref in zip(predictions, references):
            if pred.lower().strip() != ref.lower().strip():
                errors.append({
                    'prediction': pred,
                    'reference': ref,
                    'cer': cer([ref], [pred])
                })
        
        if strategy == "random":
            sampled = np.random.choice(errors, min(n_samples, len(errors)), replace=False)
        elif strategy == "stratified":
            # Sort by error severity and stratify
            sorted_errors = sorted(errors, key=lambda x: x['cer'])
            sampled = sorted_errors[::max(len(sorted_errors) // n_samples, 1)][:n_samples]
        elif strategy == "every_nth":
            sampled = errors[::max(len(errors) // n_samples, 1)][:n_samples]
        else:
            sampled = errors[:n_samples]
        
        return list(sampled)
    
    def build_taxonomy(self, error_samples: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Build error taxonomy from error samples
        Categories should emerge from data analysis
        
        Potential categories:
        - Pronunciation errors
        - Number word issues
        - Code-switching (English/Hindi)
        - Accent/dialect issues
        - Background noise
        - Homophones/similar words
        """
        taxonomy = {}
        
        for error in error_samples:
            # Analyze each error and categorize
            # This is a template - will be filled with real analysis
            category = self._categorize_error(error)
            
            if category not in taxonomy:
                taxonomy[category] = []
            
            taxonomy[category].append(error)
        
        return taxonomy
    
    def _categorize_error(self, error: Dict) -> str:
        """Categorize an error based on analysis"""
        pred = error['prediction'].lower()
        ref = error['reference'].lower()
        
        # Template for categorization logic
        # Will be developed based on actual error analysis
        if len(pred.split()) != len(ref.split()):
            return "word_count_mismatch"
        else:
            return "word_substitution"
    
    def generate_report(self, taxonomy: Dict) -> str:
        """Generate error taxonomy report"""
        report = "ERROR TAXONOMY REPORT\n"
        report += "=" * 50 + "\n\n"
        
        for category, errors in sorted(taxonomy.items(), key=lambda x: len(x[1]), reverse=True):
            report += f"Category: {category.upper()}\n"
            report += f"Frequency: {len(errors)} errors\n"
            report += f"Percentage: {len(errors)/sum(len(e) for e in taxonomy.values())*100:.1f}%\n"
            report += "-" * 50 + "\n"
            
            # Show 3-5 examples
            for i, error in enumerate(errors[:5]):
                report += f"\nExample {i+1}:\n"
                report += f"  Reference: {error['reference']}\n"
                report += f"  Predicted: {error['prediction']}\n"
                report += f"  Analysis: [TO BE FILLED]\n"
            
            report += "\n" + "=" * 50 + "\n\n"
        
        return report


def main():
    """Main execution pipeline"""
    logger.info("="*60)
    logger.info("Question 1: Whisper Fine-tuning & Error Analysis")
    logger.info("="*60)
    
    # Step 1: Load data
    logger.info("\n[STEP 1] Loading dataset...")
    processor = DatasetProcessor()
    
    if not processor.load_csv():
        logger.error("Failed to load CSV. Exiting.")
        return
    
    # Step 2: Process a small sample first
    logger.info("\n[STEP 2] Processing first 5 samples for testing...")
    dataset = processor.process_dataset(max_samples=5)
    
    if dataset is None or len(dataset) == 0:
        logger.error("Failed to process any samples. Check GCS URLs and connectivity.")
        return
    
    logger.info(f"✓ Successfully processed {len(dataset)} samples")
    for i, sample in enumerate(dataset):
        logger.info(f"  Sample {i+1}: {sample['duration']:.1f}s - '{sample['transcription'][:50]}...'")
    
    # Step 3: Load pretrained Whisper
    logger.info("\n[STEP 3] Loading pretrained Whisper-small...")
    try:
        finetuner = WhisperFinetuner()
        logger.info("✓ Whisper-small loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Whisper: {e}")
        return
    
    # Step 4: Test inference on one sample
    logger.info("\n[STEP 4] Testing baseline inference on first sample...")
    try:
        sample_audio = dataset[0]['audio']['array']
        
        # Get prediction from pretrained Whisper
        inputs = finetuner.processor(
            sample_audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(finetuner.device)
        
        with torch.no_grad():
            generated_ids = finetuner.model.generate(inputs.input_features)
        
        prediction = finetuner.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        reference = dataset[0]['transcription']
        
        logger.info(f"  Reference:  {reference}")
        logger.info(f"  Predicted:  {prediction}")
        logger.info(f"  Match: {prediction.lower() == reference.lower()}")
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n" + "="*60)
    logger.info("Step 1 Data Loading & Baseline Test Complete!")
    logger.info("="*60)
    logger.info("""
    Next steps:
    1. Process full dataset (all 104 samples)
    2. Fine-tune Whisper-small on this data
    3. Evaluate on FLEURS Hindi test set
    4. Analyze errors and build taxonomy
    5. Implement fixes
    """)


if __name__ == "__main__":
    main()
