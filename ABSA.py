import re
import nltk
from transformers import pipeline
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

# Load ABSA model
sentiment_pipeline = pipeline("sentiment-analysis", model="yangheng/deberta-v3-base-absa-v1.1")

def clean_sentence(s):
    s = s.lower()
    s = re.sub(r"\[\[ASIN:[^\]]+\]\]", "", s)
    s = re.sub(r"<br\s*/?>", " ", s)
    s = s.strip()
    return s

def analyze_review(text, aspects):
    aspect_sentiment = {}
    if not isinstance(text, str):
        return aspect_sentiment

    text = clean_sentence(text)
    
    for aspect in aspects:
        prompt = f"What do you think about the {aspect}?"
        full_text = f"{text} {prompt}"
        result = sentiment_pipeline(full_text)[0]
        label = result["label"]
        confidence = result["score"]

        sentiment_icon = "✅" if label == "Positive" else if "O" if label=="Neutral" else "❌"
        aspect_sentiment[aspect] = {
            "sentiment": sentiment_icon,
            "confidence": confidence,
            "label": label
        }

    return aspect_sentiment
