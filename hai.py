import whisper
import os

# =========================
# CONFIGURATION
# =========================

# Put your WAV file name here (same folder recommended)
audio_file = "malayalam_record2.wav"

# Check file exists (prevents errors)
if not os.path.exists(audio_file):
    print("❌ Audio file not found:", audio_file)
    print("👉 Make sure the file is in the same folder as this script.")
    exit()

# =========================
# LOAD MODEL
# =========================

print("Loading Whisper model...")

model = whisper.load_model("medium")  # best balance for Malayalam

# =========================
# TRANSCRIBE AUDIO
# =========================

print("Transcribing audio...")

result = model.transcribe(
    audio_file,
    language="ml",   # Malayalam ASR
    fp16=False,      # CPU safe (fixes warning)
    temperature=0    # improves stability
)

# =========================
# OUTPUT
# =========================

text = result["text"]

print("\n===== MALAYALAM TRANSCRIPTION =====\n")
print(text)

# =========================
# SAVE OUTPUT
# =========================

with open("output.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("\n✅ Saved to output.txt")