import pandas as pd
import nltk
from transformers import pipeline
from tqdm import tqdm
from nltk.tokenize import sent_tokenize


nltk.download('punkt')
tqdm.pandas()


def get_aspect_keywords():
    return {
        "effectiveness": [
            "pain relief", "healing time", "long-lasting", "visible results",
            "fast-acting", "effective", "works well", "noticeable improvement"
        ],
        "scent": [
            "fragrance", "natural smell", "pleasantness", "odor", "smells good", "strong scent", "aroma", "unscented"
        ],
        "ingredients": [
            "organic", "chemical-free", "active ingredients", "natural ingredients", 
            "paraben-free", "no additives", "clean ingredients", "non-toxic"
        ],
        "suitability": [
            "sensitive skin", "non-irritating", "rash-free", "allergy-friendly",
            "gentle", "safe for kids", "dermatologist-tested"
        ],
        "value": [
            "price", "quantity", "worth the cost", "affordable", "overpriced", 
            "good value", "expensive", "budget-friendly"
        ],
        "packaging": [
            "sealed", "easy to use", "leak-proof", "broken bottle", "messy", "travel-friendly", "tamper-proof"
        ],
        "texture": [
            "non-greasy", "smooth", "absorbs well", "sticky", "thick", "lightweight", "creamy"
        ],
        "application": [
            "easy to apply", "absorbs quickly", "messy to use", "instructions", "daily use", "frequency"
        ],
        "side effects": [
            "itching", "burning", "irritation", "breakout", "allergic reaction", 
            "no side effects", "skin peel"
        ],
        "longevity": [
            "lasts long", "short lifespan", "expires soon", "stable formula", "retains quality"
        ],
        "brand trust": [
            "trusted brand", "reliable", "well-known", "reputable", "popular brand", "consistent quality"
        ],
        "customer service": [
            "fast delivery", "replacement", "refund", "support", "damaged product", "return policy"
        ]
    }

sentiment_pipeline = pipeline("sentiment-analysis", model="yangheng/deberta-v3-base-absa-v1.1", device=-1)

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
