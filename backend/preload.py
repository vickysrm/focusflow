import os

os.environ["HF_HOME"] = os.path.join(os.getcwd(), "model_cache")

from sentence_transformers import SentenceTransformer
from transformers import pipeline
from faster_whisper import WhisperModel

def preload_models():
    print("Downloading models during build phase to prevent Render timeout...")
    
    print("1/3 Loading SentenceTransformer...")
    SentenceTransformer("all-MiniLM-L6-v2")
    
    print("2/3 Loading Zero-Shot Classifier...")
    pipeline("text-classification", model="typeform/distilbert-base-uncased-mnli", device=-1)
    
    print("3/3 Loading Whisper Model...")
    # faster-whisper saves to the HF cache by default
    WhisperModel("base.en", device="cpu", compute_type="int8")
    
    print("All models successfully cached!")

if __name__ == "__main__":
    preload_models()
