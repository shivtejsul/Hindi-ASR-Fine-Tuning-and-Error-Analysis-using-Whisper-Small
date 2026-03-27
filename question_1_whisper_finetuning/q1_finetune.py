"""
Question 1: Fine-tuning Whisper on Hindi ASR Data

This script fine-tunes Whisper-small on the Hindi training dataset
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset, DatasetDict
from jiwer import wer, cer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from q1_baseline import SyntheticDatasetProcessor, WhisperBaseline, ErrorAnalyzer


class WhisperFinetuner:
    """Fine-tune Whisper on Hindi data"""
    
    def __init__(self, model_name: str = "openai/whisper-tiny", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device)
        
        # Set language (Hindi)
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="hi", task="transcribe")
        
        logger.info(f"✓ Model initialized for fine-tuning on {device}")
    
    def prepare_dataset(self, dataset: Dataset, split_ratio: float = 0.8) -> DatasetDict:
        """
        Prepare dataset for fine-tuning
        - Process audio features
        - Tokenize transcriptions
        - Split into train/test (adjusted for small dataset)
        """

        def prepare_sample(batch):
            """Prepare individual sample"""
            audio = batch["audio"]

            # Process audio
            inputs = self.processor(
                audio["array"],
                sampling_rate=audio["sampling_rate"],
                return_tensors="pt"
            )

            # Tokenize text (target)
            with self.processor.as_target_processor():
                labels = self.processor.tokenizer(batch["transcription"], return_tensors="pt")

            # Create batch
            return {
                "input_features": inputs.input_features[0],
                "labels": labels.input_ids[0]
            }

        # Process dataset
        logger.info("Preparing dataset...")
        processed_dataset = dataset.map(
            prepare_sample,
            batched=False,
            remove_columns=dataset.column_names,
            num_proc=1
        )

        # For small datasets, use all data for training and create a small test set
        if len(processed_dataset) <= 5:
            # Use all for training, duplicate some for testing
            train_dataset = processed_dataset
            test_dataset = processed_dataset.select(range(min(2, len(processed_dataset))))
        else:
            # Normal split
            split = processed_dataset.train_test_split(test_size=max(0.2, 2/len(processed_dataset)), seed=42)
            train_dataset = split["train"]
            test_dataset = split["test"]

        logger.info(f"✓ Prepared {len(train_dataset)} training samples, {len(test_dataset)} test samples")

        return DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })
        
        return DatasetDict({
            "train": split["train"],
            "test": split["test"]
        })
    
    def compute_metrics(self, pred):
        """Compute WER and CER metrics during training"""
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # Replace -100 with pad_token_id
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        
        # Decode predictions and labels
        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)
        
        # Calculate metrics
        wer_score = wer(label_str, pred_str)
        cer_score = cer(label_str, pred_str)
        
        return {
            "wer": wer_score,
            "cer": cer_score
        }
    
    def train(self, training_args: Optional[Dict] = None, num_train_epochs: int = 3):
        """
        Fine-tune the model
        
        Args:
            training_args: Custom training arguments
            num_train_epochs: Number of training epochs
        """
        
        # Load and prepare dataset
        processor = SyntheticDatasetProcessor()
        dataset = processor.load_dataset()
        prepared_dataset = self.prepare_dataset(dataset)
        
        # Setup training arguments (adjusted for small dataset)
        if training_args is None:
            training_args = Seq2SeqTrainingArguments(
                output_dir="./models/whisper_hi_finetune",
                per_device_train_batch_size=2,  # Small batch size for small dataset
                per_device_eval_batch_size=2,
                gradient_accumulation_steps=4,  # Accumulate gradients
                learning_rate=1e-5,
                warmup_steps=0,  # No warmup for small dataset
                num_train_epochs=num_train_epochs,
                weight_decay=0.01,
                save_total_limit=2,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="wer",
                greater_is_better=False,
                push_to_hub=False,
                logging_steps=1,  # Log every step for small dataset
                save_steps=10,  # Save frequently
            )
        
        # Create trainer
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=self.model,
            train_dataset=prepared_dataset["train"],
            eval_dataset=prepared_dataset["test"],
            data_collator=self._data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )
        
        # Train
        logger.info("Starting fine-tuning...")
        trainer.train()
        
        # Save model
        save_path = "./models/whisper_hi_finetuned_final"
        logger.info(f"Saving model to {save_path}...")
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
        
        return trainer
    
    def _data_collator(self, features):
        """Custom data collator for Seq2Seq"""
        import torch

        # Stack input features
        input_features = [torch.tensor(feature["input_features"]) for feature in features]
        input_features = torch.stack(input_features)

        # Stack labels
        labels = [torch.tensor(feature["labels"]) for feature in features]
        labels = torch.stack(labels)

        return {
            "input_features": input_features,
            "labels": labels
        }
    
    def evaluate_finetuned(self, test_dataset: Dataset, baseline_results: Dict = None):
        """
        Evaluate fine-tuned model and compare with baseline
        """
        logger.info("Evaluating fine-tuned model...")
        
        # Get predictions
        whisper = WhisperBaseline(device=self.device)
        whisper.model = self.model
        results = whisper.evaluate(test_dataset)
        
        # Compare with baseline if provided
        if baseline_results:
            logger.info("\n" + "="*70)
            logger.info("BASELINE vs FINE-TUNED COMPARISON")
            logger.info("="*70)
            logger.info(f"Baseline WER:      {baseline_results['wer']:.4f}")
            logger.info(f"Fine-tuned WER:    {results['wer']:.4f}")
            logger.info(f"Improvement:       {baseline_results['wer'] - results['wer']:.4f} ↓")
            logger.info(f"Relative Improvement: {(baseline_results['wer'] - results['wer'])/baseline_results['wer']*100:.1f}%")
            logger.info("="*70 + "\n")
        
        return results


def main():
    """Main fine-tuning pipeline"""
    logger.info("="*70)
    logger.info("Question 1: Whisper Fine-tuning Pipeline")
    logger.info("="*70)
    
    # Step 1: Get baseline
    logger.info("\n[STEP 1] Getting baseline performance...")
    processor = SyntheticDatasetProcessor()
    dataset = processor.load_dataset()
    baseline = WhisperBaseline()
    baseline_results = baseline.evaluate(dataset)
    
    logger.info(f"✓ Baseline WER: {baseline_results['wer']:.4f}")
    
    # Step 2: Fine-tune
    logger.info("\n[STEP 2] Starting fine-tuning...")
    finetuner = WhisperFinetuner()
    
    try:
        finetuner.train(num_train_epochs=3)
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        logger.info("Note: Fine-tuning requires HF Trainer setup. See comments in code.")
        return
    
    # Step 3: Evaluate and compare
    logger.info("\n[STEP 3] Evaluating fine-tuned model...")
    finetuned_results = finetuner.evaluate_finetuned(dataset, baseline_results)
    
    # Step 4: Error analysis
    logger.info("\n[STEP 4] Analyzing errors...")
    analyzer = ErrorAnalyzer()
    errors = analyzer.find_errors(finetuned_results['predictions'], finetuned_results['references'])
    
    if errors:
        sampled_errors = analyzer.sample_errors(errors, n_samples=min(25, len(errors)))
        report = analyzer.build_error_report(sampled_errors)
        
        # Save report
        os.makedirs("results", exist_ok=True)
        with open("results/q1_finetune_error_analysis.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"\n✓ Error report saved to results/q1_finetune_error_analysis.txt")
    
    logger.info("\n" + "="*70)
    logger.info("FINE-TUNING COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
