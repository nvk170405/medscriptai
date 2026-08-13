"""Test the EasyOCR + rule-based entity extractor predictor."""
import sys
sys.path.append('src')

from pathlib import Path
from medscript.inference.predictor import MedScriptPredictor
import numpy as np
from PIL import Image, ImageDraw, ImageFont

print("=== Initializing EasyOCR + Rule-based Entity Extractor ===")
predictor = MedScriptPredictor(device="cpu")

# Create a test image with clear printed text
img = Image.new("RGB", (800, 200), color=(255, 255, 255))
draw = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype("arial.ttf", 28)
except Exception:
    font = ImageFont.load_default()

draw.text((20, 20), "Amoxicillin 500mg", fill=(0, 0, 0), font=font)
draw.text((20, 60), "Take 1 tablet TID", fill=(0, 0, 0), font=font)
draw.text((20, 100), "for 7 days", fill=(0, 0, 0), font=font)
draw.text((20, 140), "Paracetamol 650mg SOS", fill=(0, 0, 0), font=font)

print("\n=== Running inference on test image ===")
result = predictor.predict(img, run_ner=True)

print(f"\nTranscription: '{result.transcription}'")
print(f"Confidence: {sum(result.word_confidences)/len(result.word_confidences):.1%}" if result.word_confidences else "")

print(f"\nEntities found: {len(result.entities)}")
for e in result.entities:
    etype = e.get('type', '?')
    evalue = e.get('value', '?')
    econf = e.get('confidence', 0)
    print(f"  {etype:15s} -> {evalue:30s} (conf: {econf:.2f})")

if result.transcription.strip() and result.entities:
    print("\nSUCCESS - OCR + Entity Extraction working!")
elif result.transcription.strip():
    print("\nPARTIAL - OCR working, entities empty")
else:
    print("\nFAILURE - no output")
