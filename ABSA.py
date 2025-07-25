import pandas as pd
import nltk
from transformers import pipeline
from tqdm import tqdm
from nltk.tokenize import sent_tokenize


nltk.download('punkt')
tqdm.pandas()
def get_aspect_keywords():
    return {
        "performance": [
            "grip", "bounce", "speed", "accuracy", "control", "swing", "responsiveness",
            "power", "balance", "stability", "spin", "flight", "trajectory"
        ],
        "durability": [
            "long-lasting", "durable", "strong", "tough", "robust", "resilient", 
            "wear and tear", "cracks", "breakage", "holds up well"
        ],
        "comfort": [
            "comfortable", "cushioning", "padding", "breathable", "ventilated",
            "soft", "sweat-resistant", "blister-free", "ergonomic"
        ],
        "fit_and_size": [
            "fit", "true to size", "tight", "loose", "snug", "perfect fit", 
            "adjustable", "size runs small", "size runs large"
        ],
        "material_quality": [
            "rubber", "carbon fiber", "leather", "synthetic", "plastic", "foam",
            "steel", "metal", "mesh", "material quality"
        ],

        "ease_of_use": [
            "easy to use", "easy to assemble", "user-friendly", "setup", "simple", 
            "instructions", "quick install", "intuitive"
        ],
        "portability": [
            "lightweight", "easy to carry", "portable", "compact", "foldable", 
            "fits in bag", "travel-friendly"
        ],
        "packaging": [
            "well packed", "damaged box", "protective packaging", "sealed", 
            "safe delivery", "secure packaging"
        ],
        "value": [
            "worth the price", "affordable", "cheap quality", "expensive", 
            "overpriced", "great deal", "value for money"
        ],
        "maintenance": [
            "easy to clean", "low maintenance", "machine washable", "rust-free", 
            "wipes clean", "easy to store", "requires upkeep"
        ],

        "customer_service": [
            "support", "refund", "return policy", "replacement", "fast delivery", 
            "helpful staff", "response time", "damaged on arrival"
        ]
    }"""depending on the category of products"""

sentiment_pipeline = pipeline("sentiment-analysis", model="yangheng/deberta-v3-base-absa-v1.1")

def analyze_review(text, aspect_keywords):
    aspect_sentiment = {}
    if not isinstance(text, str):
        return aspect_sentiment
    sentences = sent_tokenize(text.lower())
    for aspect, keywords in aspect_keywords.items():
        matched = [s for s in sentences if any(k in s for k in keywords)]
        if matched:
            combined = " ".join(matched)[:512]
            result = sentiment_pipeline(combined)[0]['label']
            symbol = '✅' if result == 'Positive' else '❌'
            aspect_sentiment[aspect] = {
                'sentiment': symbol,
                'evidence': matched
            }
    return aspect_sentiment


aspect_keywords = get_aspect_keywords()
df3['aspect_sentiment1'] = df3['text'].progress_apply(lambda x: analyze_review(x, aspect_keywords))
