from feature_extraction import extract_mfcc

# Use ONE real .wav file path (raw string)
wav_path = r"data\RAVDESS\Actor_01\03-01-01-01-01-01-01.wav"

mfcc = extract_mfcc(wav_path)
print("MFCC shape:", mfcc.shape)
