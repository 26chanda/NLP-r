import json
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import LatentDirichletAllocation

# Load the JSON file
with open('questions.json', 'r') as f:
    question_flow = json.load(f)

# Function to handle the question flow
def ask_questions(question_id, question_flow):
    question = question_flow['questions'][question_id]
    print(question['question'])
    
    for option, details in question['options'].items():
        print(f"{option}. {details['text']}")
    
    choice = input("Choose an option: ").strip().lower()
    
    if choice in question['options']:
        next_question = question['options'][choice]['nextQuestion']
        if next_question:
            ask_questions(next_question, question_flow)
        else:
            print("End of questions.")
    else:
        print("Invalid choice. Please try again.")
        ask_questions(question_id, question_flow)

# Start the question flow from the first question
ask_questions("q1", question_flow)

# Assuming you have gathered user input from the questions, 
# map user responses to model predictions
# This is a placeholder for actual data collection from user responses
user_responses = " ".join([])#"answer option given"])

# Tokenize the user responses for BERT model input
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare user input as tokenized text
user_encoding = tokenizer(user_responses, 
                          add_special_tokens=True, 
                          max_length=512, 
                          padding='max_length', 
                          truncation=True, 
                          return_attention_mask=True, 
                          return_tensors='pt')

# Load a pre-trained BERT model
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Example BERT classifier from previous code
class BertClassifier(torch.nn.Module):
    def __init__(self, model):
        super(BertClassifier, self).__init__()
        self.model = model
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(self.model.config.hidden_size, 50)  # Assuming 50 classes/tags

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

# Initialize model
classifier_model = BertClassifier(bert_model)

# Assuming model is trained, move it to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier_model.to(device)

# Pass user input through the classifier
with torch.no_grad():
    user_input_ids = user_encoding['input_ids'].to(device)
    attention_mask = user_encoding['attention_mask'].to(device)
    logits = classifier_model(user_input_ids, attention_mask)

    predicted_tag = logits.argmax(-1).item()

# Print or store the predicted tag
print(f"Predicted Tag for the user: {predicted_tag}")


lda_model = LatentDirichletAllocation(n_components=10, max_iter=10, learning_method='online')
similarities = classifier_model.cosine_similarity(user_input_ids, attention_mask).cpu().numpy()

lda_model.fit(similarities.reshape(-1, 1))

topic_distribution = lda_model.transform(similarities.reshape(-1, 1))


suggested_topics = topic_distribution.argmax(axis=1)

print(f"Suggested Topics based on user responses: {suggested_topics}")
