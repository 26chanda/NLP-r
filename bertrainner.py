import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the new training dataset
new_train_df = pd.read_csv('synthetic_dataset.csv')

# Ensure the dataset has 'text' and 'label' columns
new_train_texts = new_train_df['text'].tolist()
new_train_labels = new_train_df['label'].tolist()

# Create a BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Set the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load a pre-trained BERT model (without classification head) to generate embeddings
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=7)
bert_model.to(device)
bert_model.eval()

def generate_embeddings(text_list):
    # Tokenize and encode the input text
    encodings = tokenizer.batch_encode_plus(text_list, 
                                             add_special_tokens=True, 
                                             max_length=512, 
                                             padding='max_length', 
                                             truncation=True, 
                                             return_attention_mask=True, 
                                             return_tensors='pt')
    
    # Move the encodings to the device (GPU or CPU)
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    
    # Generate embeddings using the BERT model
    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        embeddings = outputs.logits  # Use `logits` for sequence classification
        embeddings = embeddings[:, 0, :]  # Assuming we want the embeddings from the first token (CLS)
    
    return embeddings

# Generate embeddings for the new training data
new_train_embeddings = generate_embeddings(new_train_texts)

# Example user profile to recommend jobs for
user_profile_text = "Experienced data scientist with expertise in machine learning and data analysis."

# Generate embeddings for the user profile
user_profile_embedding = generate_embeddings([user_profile_text])

# Compute cosine similarity between user profile and all job descriptions
similarity_scores = torch.cosine_similarity(user_profile_embedding, new_train_embeddings).flatten()

# Get the top N jobs with the highest similarity scores
top_n = 5
top_n_indices = similarity_scores.argsort()[-top_n:][::-1]
top_n_jobs = new_train_df.iloc[top_n_indices]

# Print the top N recommended jobs
print("Top recommended jobs for the user profile:")
print(top_n_jobs[['text', 'label']])

# Create a custom dataset class for our data
class BertDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)
new_train_encodding=[]
# Creating data loaders for training
new_train_dataset = BertDataset(new_train_labels) 
new_train_loader = torch.utils.data.DataLoader(new_train_dataset, batch_size=16, shuffle=True)

# Load a pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=7)
model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Train the model
for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in new_train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(new_train_loader)}')

# Save the trained model if needed
torch.save(model.state_dict(), 'bert_classifier_model.pth')
