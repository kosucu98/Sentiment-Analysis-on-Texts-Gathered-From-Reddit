#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
import numpy as np

def preprocess_data(X, vectorizer=None, imputer=None, is_train=True):
    X_text = X['text']
    X_non_text = X.drop(columns='text')

    if is_train:
        vectorizer = TfidfVectorizer()
        X_text = vectorizer.fit_transform(X_text).toarray()
    else:
        X_text = vectorizer.transform(X_text).toarray()

    if is_train:
        imputer = SimpleImputer(strategy='mean')
        X_non_text = imputer.fit_transform(X_non_text)
    else:
        X_non_text = imputer.transform(X_non_text)

    X_combined = np.hstack((X_text, X_non_text))
    return X_combined, vectorizer, imputer

def load_data(file_path):
    data = pd.read_csv(file_path)
    features = ['text', 'lex_liwc_negemo', 'lex_liwc_Tone', 'lex_liwc_i', 'lex_liwc_Clout', 'sentiment', 
                'lex_liwc_posemo', 'lex_liwc_social', 'lex_liwc_Authentic', 'lex_liwc_function', 'lex_liwc_Dic']
    X = data[features]
    y = data['label']
    return X, y

def train_and_evaluate_model(X_train, y_train, X_test, y_test, model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{model_name} Accuracy:", accuracy_score(y_test, y_pred))
    print(f"{model_name} Classification Report:\n", classification_report(y_test, y_pred))

def main(train_file_path, test_file_path):
    X_train, y_train = load_data(train_file_path)
    X_test, y_test = load_data(test_file_path)
    X_train, vectorizer, imputer = preprocess_data(X_train, is_train=True)
    X_test, _, _ = preprocess_data(X_test, vectorizer=vectorizer, imputer=imputer, is_train=False)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine": SVC(kernel='linear', random_state=42)
    }
    for name, model in models.items():
        train_and_evaluate_model(X_train, y_train, X_test, y_test, model, name)

train_file_path = 'C:/Users/oguzh/Downloads/archive-8/dreaddit-train.csv'
test_file_path = 'C:/Users/oguzh/Downloads/archive-8/dreaddit-test.csv'
main(train_file_path, test_file_path)


# In[ ]:




