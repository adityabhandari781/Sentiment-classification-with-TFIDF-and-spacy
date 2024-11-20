import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

df = pd.read_csv('data/train.txt', delimiter=';', names=['comment', 'emotion'])
test_df = pd.read_csv('data/test.txt', delimiter=';', names=['comment', 'emotion'])

hash = {s:i for i,s in enumerate(df['emotion'].unique())}
df['emotion_num'] = df['emotion'].map(hash)
test_df['emotion_num'] = test_df['emotion'].map(hash)

X_train = df['comment']
y_train = df['emotion_num']
X_test = test_df['comment']
y_test = test_df['emotion_num']

nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct:
            filtered_tokens.append(token.lemma_)
    return ' '.join(filtered_tokens)

df['preprocessed'] = df['comment'].apply(preprocess)
test_df['preprocessed'] = test_df['comment'].apply(preprocess)

clf = Pipeline([
     ('tfidf',TfidfVectorizer()),
     ('rf', RandomForestClassifier())         
])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred)) 