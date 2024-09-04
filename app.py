from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
base_model = BertModel.from_pretrained('bert-base-uncased')

# Example train_label_mapping
train_label_mapping = {"Language":0, "Role Description":1, "Projects":2, "Experience":3, "Location":4, "Department":4, "Responsibilities":5}  

# Custom model
class BertClassifier(torch.nn.Module):
    def __init__(self, model, output_size):
        super(BertClassifier, self).__init__()
        self.model = model
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(self.model.config.hidden_size, output_size)

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

# Instantiate the model
model = BertClassifier(base_model, output_size=len(train_label_mapping))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the input data schema
class TextData(BaseModel):
    texts: list
    is_train: bool

@app.post("/predict/")
async def predict(data: TextData):
    try:
        # Tokenize input texts
        encodings = tokenizer.batch_encode_plus(
            data.texts, add_special_tokens=True, max_length=512, padding='max_length', truncation=True,
            return_attention_mask=True, return_tensors='pt'
        )
        encodings = {key: val.to(device) for key, val in encodings.items()}

        # Model inference
        model.eval()
        with torch.no_grad():
            if data.is_train:
                logits = model(encodings['input_ids'], encodings['attention_mask'])
                predictions = torch.argmax(logits, dim=1).cpu().numpy().tolist()
                return {"predictions": predictions}
            else:
                similarities = model.cosine_similarity(encodings['input_ids'], encodings['attention_mask'])
                return {"similarities": similarities.cpu().numpy().tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/topic_modeling/")
async def topic_modeling(similarities: list):
    try:
        # Convert similarities to numpy array
        similarities = np.array(similarities)

        # LDA model
        lda_model = LatentDirichletAllocation(n_components=50, max_iter=10, learning_method='online')
        lda_model.fit(similarities)
        topic_distributions = lda_model.transform(similarities)

        # Return the topic distributions
        return {"topic_distributions": topic_distributions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the BERT-based classification and topic modeling API"}
