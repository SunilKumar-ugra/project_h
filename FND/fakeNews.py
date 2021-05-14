import numpy as np
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
# from flask import jsonify

def resolve(news):
    # Importing dataset
    data = pd.read_csv(os.path.dirname(__file__)+"/fakeNewsData.csv")
    data = data[pd.notnull(data['text'])]
    data = data[pd.notnull(data['subject'])]
    data = data[pd.notnull(data['target'])]


    X_train,X_test,y_train,y_test = train_test_split(data['text'], data.target, test_size=0.2, random_state=42)

    # Vectorizing and applying TF-IDF
    from sklearn.linear_model import LogisticRegression

    pipe = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('model', LogisticRegression())])

    # Fitting the model
    model = pipe.fit(X_train, y_train)

    # Accuracy
    prediction = model.predict(X_test)
    #print("accuracy: {}%".format(round(accuracy_score(y_test, prediction) * 100, 2)))

    #######################################
    X_new = news
    # X_new = []
    # X_input = news
    # X_new = feature_extraction.generate_data_set(X_input)
    # X_new = np.array(X_new).reshape(1, -1)

    try:
        prediction = model.predict([X_new])
        print(prediction)
        if prediction == "true":
            return "Reliable news"
        else:
            return "Fake news"
    except:
        return "Not Sure"
