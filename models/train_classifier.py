import sys
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import re
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import multioutput
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(database_filepath):
    engine = create_engine('sqlite:///'+ str(database_filepath))
    df = pd.read_sql ('SELECT * FROM Messagescategories', engine)
    X = df ['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    #change to lower case format
     text = text.lower()
      # Remove punctuation characters
     text = re.sub(r"[^a-zA-Z0-9]", " ", text)
     stop_words = stopwords.words("english")
     
     #tokenize
     words = word_tokenize (text)
     
     # Reduce words to their stems
     stemmed = [PorterStemmer().stem(w) for w in words]
     
     #lemmatizing
     lemmed = [WordNetLemmatizer().lemmatize(w) for w in stemmed if w not in stop_words]
    
     return lemmed

def build_model():
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', multioutput.MultiOutputClassifier (RandomForestClassifier()))
            ])
    parameters= {
            'tfidf__use_idf':[True],
            }
    cv = GridSearchCV(pipeline, parameters)
   
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Print model results
    INPUT
    model -- required, estimator-object
    X_test -- required
    y_test -- required
    category_names = required, list of category strings
    OUTPUT
    None
    """
    # Get results and add them to a dataframe.
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))
    #results = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])
        
def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()