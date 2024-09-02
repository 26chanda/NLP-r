import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import confusion_matrix, classification_report


# Load the dataset
train_df = pd.read_csv('synthetic_dataset.csv')
test_df = pd.read_csv('testdata.csv')
print(train_df.columns)
print(test_df.columns)

# Create a BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare the data
# Convert all elements to strings before joining
train_texts = train_df[['Language', 'Role Description', 'Projects', 'Experience', 'Location', 'Department']].apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
test_texts = test_df[['Languages spoken', 'job description', 'Work experience', 'Location', 'Industry']].apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()

# Define a label mapping function
# Example label mapping (update based on your dataset)
def create_label_mapping(df, label_columns):
    unique_labels = sorted(df[label_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1).unique())
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    return label_mapping

# Create label mappings for train and test data
train_label_mapping = create_label_mapping(train_df, ['Language', 'Role Description', 'Projects', 'Experience', 'Location', 'Department', 'Responsibilities'])
test_label_mapping = create_label_mapping(test_df, ['Languages spoken', 'job description', 'Work experience', 'Location'])

# Debugging: Print the label mappings
print("Train Label Mapping:", train_label_mapping)
print("Test Label Mapping:", test_label_mapping)


# Map the labels to integers
try:
    train_labels_str = train_df[['Language', 'Role Description', 'Projects', 'Experience', 'Location', 'Department', 'Responsibilities']].apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
    train_labels = [train_label_mapping.get(label, -1) for label in train_labels_str]

    test_labels_str = test_df[['Languages spoken', 'job description', 'Work experience', 'Location']].apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
    test_labels = [test_label_mapping.get(label, -1) for label in test_labels_str]
except KeyError as e:
    print(f"KeyError: {e}. Ensure that all labels are correctly mapped.")

# Tokenize the texts
train_encodings = tokenizer.batch_encode_plus(train_texts, 
                                              add_special_tokens=True, 
                                              max_length=512, 
                                              padding='max_length', 
                                              truncation=True, 
                                              return_attention_mask=True, 
                                              return_tensors='pt')

test_encodings = tokenizer.batch_encode_plus(test_texts, 
                                             add_special_tokens=True, 
                                             max_length=512, 
                                             padding='max_length', 
                                             truncation=True, 
                                             return_attention_mask=True, 
                                             return_tensors='pt')

# Create a custom dataset class for data
class BertDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  
        return item

    def __len__(self):
        return len(self.labels)

# Creating data loaders for training and testing
train_dataset = BertDataset(train_encodings, train_labels)
test_dataset = BertDataset(test_encodings, test_labels)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load a pre-trained BERT model
model = BertModel.from_pretrained('bert-base-uncased')

# Set the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define a custom model for classification
class BertClassifier(torch.nn.Module):
    def __init__(self, model):
        super(BertClassifier, self).__init__()
        self.model = model
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(self.model.config.hidden_size, len(train_label_mapping))  # Adjust the output size

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
    
    def cosine_similarity(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        embeddings = self.model.get_input_embeddings()(input_ids)
        cosine_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        similarities = cosine_sim(pooled_output, embeddings)
        return similarities

# Creating an instance of the custom model
model = BertClassifier(model)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Train the model
for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}')

model.eval()

# Evaluate the model on the test set
test_labels_pred = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        logits = model(input_ids, attention_mask)
        labels_pred = logits.argmax(-1).detach().cpu().numpy()
        test_labels_pred.extend(labels_pred)

accuracy = accuracy_score(test_labels, test_labels_pred)
print(f'Accuracy: {accuracy:.4f}')
# Compute and print the confusion matrix
conf_matrix = confusion_matrix(test_labels, test_labels_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Print the classification report
class_report = classification_report(test_labels, test_labels_pred)
print("Classification Report:")
print(class_report)
# Topic modeling with Latent Dirichlet Allocation (LDA)
lda_model = LatentDirichletAllocation(n_components=50, max_iter=10, learning_method='online')

# Collecting training data embeddings for topic modeling
train_similarities = []
with torch.no_grad():
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        similarities = model.cosine_similarity(input_ids, attention_mask)
        train_similarities.extend(similarities.detach().cpu().numpy())

# Fit the LDA model
lda_model.fit(train_similarities)

# Transform test set similarities to topic distributions
test_similarities = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        similarities = model.cosine_similarity(input_ids, attention_mask)
        test_similarities.extend(similarities.detach().cpu().numpy())

topic_distributions = lda_model.transform(test_similarities)

# Save topic distributions for each cluster
cluster_topics = {}
for i, topic_dist in enumerate(topic_distributions):
    cluster_id = f"cluster_{i}"
    cluster_topics[cluster_id] = topic_dist
    with open(f"{cluster_id}.txt", "w") as f:
        f.write("\n".join([f"Topic {j}: {topic_dist[j]:.4f}" for j in range(topic_dist.shape[0])]))

print("Topic distributions saved to separate files.")

