import pandas as pd
import torch
from transformers import BertTokenizer
import requests
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
train_df = pd.read_csv('synthetic_dataset.csv')
test_df = pd.read_csv('testdata.csv')

# Create a BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare the data
train_texts = train_df[['Language', 'Role Description', 'Projects', 'Experience', 'Location', 'Department']].apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
test_texts = test_df[['Languages spoken', 'job description', 'Work experience', 'Location', 'Industry']].apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()

# Define label mapping functions and create label mappings
def create_label_mapping(df, label_columns):
    unique_labels = sorted(df[label_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1).unique())
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    return label_mapping

train_label_mapping = create_label_mapping(train_df, [ 'Projects', 'Experience', 'Location', 'Department', 'Responsibilities'])
test_label_mapping = create_label_mapping(test_df, [ 'job description', 'Work experience', 'Location'])

# Map the labels to integers
train_labels_str = train_df[['Projects', 'Experience',  'Department', 'Responsibilities']].apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
train_labels = [train_label_mapping.get(label, -1) for label in train_labels_str]

test_labels_str = test_df[['job description', 'Work experience']].apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
test_labels = [test_label_mapping.get(label, -1) for label in test_labels_str]

# Prepare API request data
train_data = {
    "texts": train_texts,
    "is_train": True
}

test_data = {
    "texts": test_texts,
    "is_train": False
}

# URL of the running FastAPI server
predict_url = "http://127.0.0.1:8000/predict/"

# Sending POST request for training data
train_response = requests.post(predict_url, json=train_data)
train_predictions = train_response.json().get("predictions", [])

# Sending POST request for test data
test_response = requests.post(predict_url, json=test_data)
test_similarities = test_response.json().get("similarities", [])

# Evaluate the model on the test set using predictions
test_labels_pred = train_predictions  # Replace this with the actual test predictions if necessary
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

# For topic modeling
topic_modeling_url = "http://127.0.0.1:8000/topic_modeling/"
response = requests.post(topic_modeling_url, json={"similarities": test_similarities})
topic_distributions = response.json().get("topic_distributions", [])

# Save topic distributions for each cluster
for i, topic_dist in enumerate(topic_distributions):
    cluster_id = f"cluster_{i}"
    with open(f"{cluster_id}.txt", "w") as f:
        f.write("\n".join([f"Topic {j}: {topic_dist[j]:.4f}" for j in range(len(topic_dist))]))

print("Topic distributions saved to separate files.")
