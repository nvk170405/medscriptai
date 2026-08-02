import sys
import os
sys.path.append('src')

from medscript.inference.predictor import MedScriptPredictor
from api.core.config import settings

def main():
    print("Loading predictor...")
    predictor = MedScriptPredictor(
        checkpoint_path=settings.model_checkpoint_path,
        device="cpu"
    )
    
    # Create a dummy image
    import numpy as np
    dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    print("Testing with 640x640 image...")
    predictor.target_height = 640
    predictor.target_width = 640
    res = predictor.predict(dummy_img, run_ner=False)
    print(f"Transcription (640x640): '{res.transcription}'")
    
    print("Testing with 960x1280 image...")
    predictor.target_height = 960
    predictor.target_width = 1280
    res2 = predictor.predict(dummy_img, run_ner=False)
    print(f"Transcription (960x1280): '{res2.transcription}'")

if __name__ == "__main__":
    main()
