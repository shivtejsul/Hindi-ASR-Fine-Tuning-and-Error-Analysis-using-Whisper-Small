# Quick Start Guide - Hindi ASR Assignment

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies
```bash
cd d:\josh\hindi_asr_assignment
pip install -r requirements.txt
```

### Step 2: Configure Python Environment
In VS Code:
- Open any `.py` file in the assignment folder
- Configure your Python environment (CUDA recommended for Whisper training)

### Step 3: Understand the 4 Questions

#### Question 1: Whisper Fine-tuning (Hardest, Most Time)
- **What:** Fine-tune OpenAI Whisper-small on Hindi audio
- **File:** `question_1_whisper_finetuning/q1_solution.py`
- **Key Deliverables:**
  - Error taxonomy with concrete examples
  - Implement at least 1 fix (e.g., better handling of numbers)
  - Show before/after WER improvement
- **Time:** ~3-4 hours

#### Question 2: Cleanup Pipeline (Medium)
- **What:** Build pipeline to clean raw ASR output
- **File:** `question_2_cleanup_pipeline/q2_solution.py`
- **Two Operations:**
  1. Number normalization: `दो` → `2`, `तीन सौ चौवन` → `354`
  2. English word detection: Mark English words in Hindi text
- **Key Challenge:** Handle idioms (`दो-चार बातें` should NOT become `2-4 बातें`)
- **Time:** ~1.5-2 hours

#### Question 3: Spelling Classification (Medium-Hard)
- **What:** Classify ~177,000 words as correctly or incorrectly spelled
- **File:** `question_3_spelling_classification/q3_solution.py`
- **Key Notes:**
  - English in Devanagari (`कं प्यूटर`) = CORRECT
  - Add confidence scores to each classification
  - Analyze wrong predictions in low-confidence bucket
- **Output:** CSV file for Google Sheets
- **Time:** ~2-2.5 hours

#### Question 4: Lattice-based WER (Conceptual)
- **What:** Design fair ASR evaluation allowing multiple valid transcriptions
- **File:** `question_4_lattice_wer/q4_solution.py`
- **Key Insight:** 
  ```
  Same audio can have multiple valid representations:
  - "चौदह" vs "14" (same number)
  - "कताबें" vs "किताबें" (variants)
  
  Lattice captures all at each position, compare models fairly
  ```
- **Time:** ~1.5-2 hours

---

## 📋 Workflow

### For Each Question:
1. **Read** the template file (`q1_solution.py`, etc.)
2. **Understand** the main classes and their methods
3. **Implement** missing functionality (marked with `TODO` or `pass`)
4. **Test** on sample data first
5. **Document** findings and save results

### Tips:
- Use small sample datasets first (10-20 examples) before full dataset
- Test each component independently
- Save intermediate results to avoid recomputation
- Use logging liberally for debugging
- Reference `utils/helpers.py` for common operations

---

## 🎯 Key Insights

### Transcription Guideline (VERY IMPORTANT)
English words spoken in Hindi are transcribed in **Devanagari script**:
- "computer" spoken → written as "कं प्यूटर" ✓ CORRECT
- NOT written as "computer" in Roman script
- This is by design, not a mistake

### Common Error Categories (From Data)
1. **Pronunciation variants** (regional accents)
2. **Number words** (दस vs 10)
3. **Code-switching** (Hindi + English)
4. **Homophones** (similar-sounding words)
5. **Fast speech** (words merged)
6. **Background noise** (causing misheard words)

### Data Access
All data is on Google Cloud Storage (GCS):
```
https://storage.googleapis.com/upload_goai/{user_id}/{recording_id}_transcription.json
https://storage.googleapis.com/upload_goai/{user_id}/{recording_id}.wav
https://storage.googleapis.com/upload_goai/{user_id}/{recording_id}_metadata.json
```

Use the `DatasetProcessor` class in `q1_solution.py` to fetch and process.

---

## 📊 Expected Outputs

### Q1: Whisper Fine-tuning
✓ WER scores (baseline vs fine-tuned)
✓ Error taxonomy report (5-7 categories)
✓ 25+ error examples with analysis
✓ Before/after results for implemented fix

### Q2: Cleanup Pipeline
✓ 4-5 correct conversion examples
✓ 2-3 tricky edge case examples with reasoning
✓ Test results showing where normalization helps/hurts

### Q3: Spelling Classification
✓ CSV with: word | classification | confidence | reason
✓ Count of correctly spelled words (~60-70% likely)
✓ Accuracy of low-confidence bucket
✓ 1-2 unreliable word categories with explanation

### Q4: Lattice WER
✓ Lattice design document (pseudocode)
✓ WER comparison (traditional vs lattice)
✓ Show models no longer unfairly penalized

---

## 🔧 Helpful Commands

```bash
# Check Python version
python --version

# List installed packages
pip list | grep -E "(torch|transformers|datasets|jiwer)"

# Run a specific question
python question_1_whisper_finetuning/q1_solution.py

# Run tests (if added)
pytest -v

# Check directory size (useful for large datasets)
du -sh data/
```

---

## 📖 References

- **Transformers:** https://huggingface.co/docs/transformers/
- **OpenAI Whisper:** https://github.com/openai/whisper
- **FLEURS Dataset:** https://huggingface.co/datasets/google/fleurs
- **jiwer (WER calculation):** https://github.com/jitsi/jiwer

---

## 🚨 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch size, use CPU, enable gradient checkpointing |
| Can't download from GCS | Check internet, verify URL format, use helper functions |
| Hindi characters display wrong | Use UTF-8 encoding everywhere |
| Whisper training too slow | Use smaller model (tiny/small), fewer epochs, smaller dataset |
| Confidence metrics unclear | Review actual error patterns manually |

---

## ✅ Submission Checklist

Before submitting (per question):

- [ ] Code is well-commented
- [ ] All TODO items handled
- [ ] Results saved to files
- [ ] Deliverables match specification
- [ ] Examples/findings documented
- [ ] No hardcoded paths (use relative paths)
- [ ] Dependencies listed in requirements.txt

---

## Next: Start with Question 1! 🎬

1. Open `question_1_whisper_finetuning/q1_solution.py`
2. Review the `DatasetProcessor` and `WhisperFinetuner` classes
3. Implement the `main()` function with actual data
4. Start with 5-10 audio samples to test
5. Expand to full dataset once working

Good luck! 🎉
