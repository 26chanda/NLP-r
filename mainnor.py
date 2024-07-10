import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import is_classifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import accuracy_score
from sklearn.naive_bayes import MuiltinomialNB

documents=[]
tokenized_docs=[]
for doc in documents:
    tokens = word_tokenize(doc)
    tokenized_docs.append(tokens)

stemmed_docs = []
lemmatizer = WordNetLemmatizer()
for tokens in tokenized_docs:
    stemmed_tokens= [lemmatizer.lemmatize(token)]
    for token in tokens:
        stemmed_docs.append(stemmed_tokens)
        
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(stemmed_docs)
Y = ()#labels

is_classifier 
clf= MuiltinomialNB()
clf.fit(X,Y)

test_docs = []# list of document
test_labels = []#list of test label
test_X = vectorizer.transform(test_docs)
test_Y_pred = clf.predict(test_X)
accuracy = accuracy_score(test_labels, test_Y_pred)
print  ('Accuracy:',accuracy)



