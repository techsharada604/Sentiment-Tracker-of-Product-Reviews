import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
model.eval()  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def chunk_review(text, max_len=512):
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_len - 2):  
        token_chunk = tokens[i:i + max_len - 2]
        if token_chunk:  
            chunk = tokenizer.convert_tokens_to_string(token_chunk)
            chunks.append(chunk)
    return chunks


def predict_sentiment(text):
    if not isinstance(text, str) or text.strip() == "":
        return np.nan

    chunks = chunk_review(text)
    scores = []

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = torch.softmax(outputs.logits, dim=1)
        scores.append(probs.cpu().numpy())

    avg_probs = np.mean(scores, axis=0)
    return int(np.argmax(avg_probs) + 1) 
df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)
tqdm.pandas()
df["bert_sentiment1"] = df["cleaned_text"].progress_apply(predict_sentiment)
