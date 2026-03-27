# Hindi ASR Assignment - AI Researcher Intern (Speech & Audio)

## Project Structure

```
hindi_asr_assignment/
├── question_1_whisper_finetuning/     # Q1: Whisper fine-tuning & error analysis
├── question_2_cleanup_pipeline/       # Q2: Number normalization & English word detection
├── question_3_spelling_classification/# Q3: Spelling error classification
├── question_4_lattice_wer/           # Q4: Lattice-based WER evaluation
├── data/                              # Raw and processed data
├── utils/                             # Shared utility functions
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Questions Overview

### Question 1: Whisper Fine-tuning & Error Analysis
- Preprocess ~10 hours of Hindi ASR training data
- Fine-tune Whisper-small model
- Evaluate on FLEURS Hindi test set
- Build error taxonomy with examples
- Implement fixes for top 3 error types

### Question 2: ASR Output Cleanup Pipeline
- Number normalization (दो → 2, तीन सौ चौवन → 354, etc.)
- English word detection in Hindi text
- Handle edge cases intelligently
- Show before/after examples

### Question 3: Spelling Error Classification
- Classify ~177,000 unique words as correct/incorrect spelling
- Add confidence scores (high/medium/low)
- Analyze low-confidence bucket accuracy
- Identify unreliable word categories

### Question 4: Lattice-based WER Evaluation
- Design lattice structure for multiple valid transcriptions
- Handle insertions, deletions, substitutions fairly
- Decide when to trust model agreement over reference
- Compare lattice-based WER vs traditional WER

## Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure your Python environment with the workspace

3. Start with Question 1 template

## Data Notes

- Dataset URL pattern: `https://storage.googleapis.com/upload_goai/{user_id}/{recording_id}_transcription.json`
- Audio: `https://storage.googleapis.com/upload_goai/{user_id}/{recording_id}.wav`
- Metadata: `https://storage.googleapis.com/upload_goai/{user_id}/{recording_id}_metadata.json`

## Key Resources

- Transformers documentation: https://huggingface.co/docs/transformers/
- OpenAI Whisper: https://github.com/openai/whisper
- FLEURS dataset: https://huggingface.co/datasets/google/fleurs

## Transcription Guidelines

English words spoken in conversations are transcribed in Devanagari script:
- "computer" → "कं प्यूटर" (correct)
- Not an error; don't penalize the model for this

---

Start with `question_1_whisper_finetuning/` first.
