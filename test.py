import pandas as pd
import os
from IPython.display import Audio, display
import tempfile

df = pd.read_parquet("sample_qa_BBQ_bi_gender.parquet")

print("=" * 60)
print("Dataset Summary")
print("=" * 60)
print(f"Shape: {df.shape}")
print("\nColumns:\n", df.columns.to_list())
print("\nData Types:\n", df.dtypes)
print("\nNull Values:\n", df.isnull().sum())
print("\nPreview:\n", df.head(2))

print("\nAudio and corresponding text:\n")

for i in range(3):
    row = df.iloc[i]
    audio_data = row['audio']
    
    print(f"\nRow {i+1}")
    print(f"Speaker: {row['speaker']}")
    print(f"Context: {row['context']}")
    print(f"Question: {row['question']}")
    print(f"Answers: [{row['ans0']}, {row['ans1']}, {row['ans2']}]")
    print(f"Label: {row['label']}")

    if isinstance(audio_data, dict) and 'bytes' in audio_data:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            tmp_audio.write(audio_data['bytes'])
            tmp_audio_path = tmp_audio.name
        
        display(Audio(filename=tmp_audio_path))
    else:
        print("Audio format not recognized")
