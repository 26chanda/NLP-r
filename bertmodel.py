import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score # type: ignore

# Load the dataset
train_df = pd.read_csv('synthetic_dataset.csv')
test_df = pd.read_csv('testdata.csv')


# Create a BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


train_texts = train_df['text']
train_labels = train_df['label']

test_texts = test_df['text']
test_labels = test_df['label']


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


# Create a custom dataset class for our data
class BertDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Creating data loaders for training and testing
train_dataset = BertDataset(train_encodings, train_labels)
test_dataset = BertDataset(test_encodings, test_labels)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load a pre-trained BERT model
model = BertModel.from_pretrained('bert-base-uncased', num_labels=8)

# Set the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define a custom model for classification
class BertClassifier(torch.nn.Module):
    def __init__(self, model):
        super(BertClassifier, self).__init__()
        self.model = model
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(self.model.config.hidden_size, 7)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        outputs = self.classifier(pooled_output)
        return outputs

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

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')

model.eval()

# Evaluate the model on the test set
test_labels_pred = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask)
        logits = outputs.detach().cpu().numpy()
        labels_pred = logits.argmax(-1)
        test_labels_pred.extend(labels_pred)

accuracy = accuracy_score(test_labels, test_labels_pred)
print(f'Accuracy: {accuracy:.4f}')