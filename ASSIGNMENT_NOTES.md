# Hindi ASR Assignment - Project Notes

## Quick Start

1. **Install dependencies:**
   ```bash
   cd d:\josh\hindi_asr_assignment
   pip install -r requirements.txt
   ```

2. **Configure Python environment**
   - Use the configure_python_environment tool
   - Ensure you have Python 3.9+
   - CUDA recommended for faster training

3. **Run individual questions:**
   ```bash
   # Question 1
   python question_1_whisper_finetuning/q1_solution.py
   
   # Question 2
   python question_2_cleanup_pipeline/q2_solution.py
   
   # Question 3
   python question_3_spelling_classification/q3_solution.py
   
   # Question 4
   python question_4_lattice_wer/q4_solution.py
   ```

## Key Concepts

### Question 1: Whisper Fine-tuning
- **Goal:** Fine-tune Whisper-small on Hindi ASR data
- **Dataset:** ~10 hours of Hindi audio + transcriptions
- **Key Steps:**
  1. Download and preprocess audio
  2. Fine-tune model on Hindi data
  3. Evaluate on FLEURS Hindi test set
  4. Build error taxonomy
  5. Implement fixes

### Question 2: Cleanup Pipeline
- **Goal:** Clean raw ASR output before downstream processing
- **Two Operations:**
  1. **Number Normalization:** दो → 2, तीन सौ चौवन → 354
  2. **English Word Detection:** Mark English words in Hindi text
- **Key Challenge:** Handle edge cases (idioms, transliterations)

### Question 3: Spelling Classification
- **Goal:** Classify ~177,000 unique words as correct/incorrect
- **Considerations:**
  - English transliterations are NOT errors (कं प्यूटर is correct)
  - Confidence scores needed
  - Analyze unreliable cases
- **Output:** CSV for Google Sheets

### Question 4: Lattice-based WER
- **Goal:** Fair ASR evaluation allowing multiple valid transcriptions
- **Key Idea:** Use lattice instead of rigid string
  - Position 1: [उसने]
  - Position 2: [चौदह, 14]
  - Position 3: [कताबें, किताबें]
  - Position 4: [खरीदीं, खरीदी]
- **Benefits:** Don't penalize valid variants

## Data Sources

All data is on Google Cloud Storage (GCS):
- **Base URL:** `https://storage.googleapis.com/upload_goai/{user_id}/`
- **Audio:** `{recording_id}.wav`
- **Transcription:** `{recording_id}_transcription.json`
- **Metadata:** `{recording_id}_metadata.json`

## Important Notes

1. **Transcription Guidelines:**
   - English words are transcribed in Devanagari
   - "computer" → "कं प्यूटर" (NOT an error)
   - "interview" → "इंटरव्यू"

2. **FLEURS Dataset:**
   - Test set: https://huggingface.co/datasets/google/fleurs
   - Use Hindi portion for evaluation

3. **Error Categories to Look For:**
   - Pronunciation variants (regional accents)
   - Number words vs digits
   - Code-switching (Hindi + English)
   - Homophones/similar words
   - Background noise
   - Fast speech

## Deliverables Checklist

### Q1
- [ ] Preprocessing steps documented
- [ ] Fine-tuned model saved
- [ ] WER metrics on FLEURS
- [ ] 25+ error samples (systematically sampled)
- [ ] Error taxonomy with 3-5 examples per category
- [ ] Top 3 error type fixes proposed
- [ ] At least 1 fix implemented with before/after

### Q2
- [ ] Number normalization working
- [ ] 4-5 before/after examples with correct conversions
- [ ] 2-3 edge case examples with reasoning
- [ ] English word detection with tagging
- [ ] Examples showing where it helps/hurts

### Q3
- [ ] Word classification complete
- [ ] CSV with word + classification + confidence
- [ ] Total count of correctly spelled words
- [ ] 40-50 low confidence words reviewed
- [ ] Accuracy analysis of low confidence bucket
- [ ] 1-2 unreliable word categories identified

### Q4
- [ ] Lattice design documented (theory + pseudocode)
- [ ] Lattice construction implemented
- [ ] Model agreement detection working
- [ ] WER compared (traditional vs lattice-based)
- [ ] Show improvement for models unfairly penalized

## Testing & Validation

Before submitting:
1. Test on sample data first
2. Check memory usage (especially for Whisper training)
3. Validate output formats match requirements
4. Document any assumptions or deviations
5. Save all results reproducibly

## Tips

- Use the `utils/helpers.py` module for common operations
- Add logging liberally for debugging
- Save intermediate results (don't recompute)
- Test on small datasets first, then scale
- Document your findings in notebooks/reports
