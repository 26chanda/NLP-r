from transformers import BertTokenizer, BertModel
import torch
from sklearn.cluster import KMeans
import numpy as np

# Load the BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Encode texts to BERT embeddings
def get_bert_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the mean of the token embeddings as the sentence embedding
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# Get embeddings for the dataset
texts = train_df['text'].tolist()
embeddings = get_bert_embeddings(texts).numpy()

# Apply KMeans clustering
num_clusters = 5  # Number of topics
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(embeddings)

# Print the topic for each document
for i, label in enumerate(kmeans.labels_):
    print(f"Document {i} is in topic {label}")
