"""
Create synthetic Hindi ASR dataset for testing
This generates random Hindi audio and transcriptions for development
"""

import numpy as np
import soundfile as sf
import json
import os
from pathlib import Path
import pandas as pd

def create_synthetic_dataset(n_samples=10, output_dir="data/synthetic"):
    """Create synthetic audio and transcription data"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample Hindi sentences
    hindi_sentences = [
        "नमस्ते मेरा नाम राज है",
        "आज का मौसम बहुत अच्छा है",
        "मुझे कंप्यूटर प्रोग्रामिंग पसंद है",
        "भारत एक बहुत सुंदर देश है",
        "क्या आप मेरी मदद कर सकते हैं",
        "मैं स्कूल जाने वाला हूँ",
        "चाय पीना मुझे पसंद है",
        "यह किताब बहुत रोचक है",
        "मेरा परिवार बहुत प्यारा है",
        "मुझे गाना गाना पसंद है",
    ]
    
    data = []
    
    for i in range(n_samples):
        # Generate synthetic audio (random noise as placeholder)
        sample_rate = 16000
        duration = np.random.uniform(3, 8)  # 3-8 seconds
        num_samples = int(duration * sample_rate)
        
        # Create realistic-ish audio (low-frequency noise)
        t = np.linspace(0, duration, num_samples)
        audio = np.sin(2 * np.pi * 100 * t) * 0.1
        audio += np.random.normal(0, 0.02, num_samples)
        audio = np.clip(audio, -1, 1)
        
        # Save audio
        audio_path = os.path.join(output_dir, f"audio_{i:04d}.wav")
        sf.write(audio_path, audio, sample_rate)
        
        # Create transcription
        transcript = hindi_sentences[i % len(hindi_sentences)]
        transcript_path = os.path.join(output_dir, f"transcription_{i:04d}.json")
        
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump({'transcription': transcript}, f, ensure_ascii=False, indent=2)
        
        data.append({
            'user_id': 100000 + i,
            'recording_id': 500000 + i,
            'language': 'hi',
            'duration': int(duration),
            'audio_path': audio_path,
            'transcript_path': transcript_path,
            'transcription': transcript
        })
        
        print(f"Created sample {i+1}/{n_samples}: {transcript[:40]}...")
    
    # Save metadata
    csv_path = os.path.join(output_dir, 'metadata.csv')
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metadata to {csv_path}")
    
    return data

if __name__ == "__main__":
    print("Creating synthetic Hindi ASR dataset...")
    create_synthetic_dataset(n_samples=10)
    print("\nDataset creation complete!")
