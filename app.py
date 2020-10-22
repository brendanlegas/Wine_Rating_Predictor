from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.externals import joblib
import joblib


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/model')
def model():
    return render_template('model.html')

@app.route('/predict', methods=['POST'])
def predict():
    nlp_model = open('wine_nlp_model2.pkl','rb')
    vectorizer = open('vectorizer2.pkl','rb')
    rfc = joblib.load(nlp_model)
    cv = joblib.load(vectorizer)
    #vectorizer = CountVectorizer()

    if request.method == 'POST':
        input_text = request.form['description']
        data = [input_text]
        vect = cv.transform(data).toarray()
        model_prediction = rfc.predict(vect)
    return render_template('top_wine_list.html', prediction = model_prediction)

@app.route('/winelist')
def winelist():
    return render_template('top_wine_list.html')

@app.route('/tryit')
def tryit():
    return render_template('tryit.html')

if __name__ == '__main__':
	app.run(debug=False)