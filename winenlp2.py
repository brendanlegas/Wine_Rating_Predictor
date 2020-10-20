#Import dependencies
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

#Read winemag csv
df = pd.read_csv('winemag-data_first150k.csv')

#Clean up dataset by dropping duplicates and nulls
clean = df[df.duplicated('description', keep=False)]
clean.dropna(subset=['description', 'points'])

#Create simplified df of just description and points
simple = clean[['description', 'points']]

#Transform method taking points as param
def simplify(points):
    if points < 82:
        return 1
    elif points >= 82 and points < 86:
        return 2 
    elif points >= 86 and points < 90:
        return 3 
    elif points >= 90 and points < 94:
        return 4 
    else:
        return 5

simple = simple.assign(rating = simple['points'].apply(simplify))

#Grabbing stopwords from nltk library
#!pip install nltk

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')

#Removing stopworks
simple['description'] = simple.description.str.replace("[^\w\s]", "").str.lower()
simple['description'] = simple['description'].apply(lambda x: ' '.
                                                    join([item for item in x.split() if item not in stop_words]))

#Description Vectorization
X = simple['description']
y = simple['rating']

#vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)
X.shape

#Splitting 80/20 train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=101)
#Using RandomForestClassifier Class to fit model to training data subset
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

#Testing the model
predictions = rfc.predict(X_test)

from sklearn.externals import joblib
joblib.dump(rfc, 'wine_nlp_model2.pkl')
joblib.dump(vectorizer, 'vectorizer2.pkl')