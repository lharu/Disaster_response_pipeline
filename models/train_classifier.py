# import libraries
import sys
import numpy as np
import pandas as pd
import pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from sqlalchemy import create_engine



def load_data(database_filepath):
    '''load data from database
    
    Parameters
    ----------
    database_filepath : string
        
    Returns
    -------
    X: np.ndarray
    Y: np.ndarray
    category_names: list
     '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('processed_data', engine)
    df = df.dropna() 
    Y = df.iloc[:,4:]
    X = df.iloc[:,1]
    categories = Y.columns
    return X, Y, categories


def tokenize(text):
     '''Tokenize the text
    
    Parameters
    ----------
    text : string
  
    Returns
    -------
    clean_tokens : list
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''Build a machine learning pipeline
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)), 
        ('tfidf', TfidfTransformer()),
    
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])


    parameters = {
            'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 1.0),
        'clf__estimator__n_estimators':[50, 100]
    }
    
    
    model = GridSearchCV(pipeline, parameters)
    return model



def evaluate_model(model, X_test, Y_test, category_names):
    '''Test the model performance
    
    Parameters
    ----------
    model : multiclassification model
    X_test: numpy.ndarray
        The test data
    Y_test: numpy.ndarray
        The test category
    category_names: list
     The category names
     
    Returns
    -------
    None
    '''
    Y_pred = model.predict(X_test)
    
    for i in range(36):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:,i], Y_pred[:,i]))




def save_model(model, model_filepath):
    ''' Save model in pickle format
    
    Parameters
    ----------
    model : a multiclassification model
        The optimized classifier
    model_filepath : string
        the database location
    
    Returns
    -------
    None

    '''
    pickle.dump(model, open(model_filepath, "wb"))



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