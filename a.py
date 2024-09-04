import requests

# URL of the running FastAPI server
url = "http://127.0.0.1:8000/predict/"

# Example input data
data = {
    "texts": ["Example text for classification"],
    "is_train": True
}

# Sending POST request
response = requests.post(url, json=data)
print(response.json())

# For topic modeling
url = "http://127.0.0.1:8000/topic_modeling/"
similarities = [...]  # Your list of similarities
response = requests.post(url, json={"similarities": similarities})
print(response.json())
