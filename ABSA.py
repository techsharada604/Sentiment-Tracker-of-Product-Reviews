import re
import nltk
from transformers import pipeline
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

# Initialize sentiment model (binary, strong general-purpose)
sentiment_pipeline = pipeline("sentiment-analysis", model="yangheng/deberta-v3-base-absa-v1.1")

def clean_sentence(s):
    s = s.lower()
    s = re.sub(r"\[\[ASIN:[^\]]+\]\]", "", s)
    s = re.sub(r"<br\s*/?>", " ", s)
    s = s.strip()
    return s

# Custom helper to group contrastive sentences
def group_with_context(sentences, i, window=1):
    start = max(0, i - window)
    end = min(len(sentences), i + window + 1)
    return " ".join(sentences[start:end])

def analyze_review(text, aspect_keywords):
    aspect_sentiment = {}
    if not isinstance(text, str):
        return aspect_sentiment

    sentences = sent_tokenize(text)
    cleaned_sentences = [clean_sentence(s) for s in sentences]

    for aspect, keywords in aspect_keywords.items():
        evidence = []
        for i, sent in enumerate(cleaned_sentences):
            if any(k in sent for k in keywords):
                context_block = group_with_context(cleaned_sentences, i)
                result = sentiment_pipeline(context_block)[0]
                label = result['label']
                confidence = result['score']

                sentiment_icon = '✅' if label == 'POSITIVE' else '❌'
                aspect_sentiment[aspect] = {
                    'sentiment': sentiment_icon,
                    'confidence': confidence,
                    'evidence': [sent]
                }
                break  

    return aspect_sentiment
