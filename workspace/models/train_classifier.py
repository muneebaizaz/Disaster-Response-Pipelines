# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
#import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report , precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.multioutput import MultiOutputClassifier




def load_data(database_filepath):
    '''
    This function loads the table from the sql database 
    by taking the database filepath as input
    Parameters:
                    database_filepath (str): filepath for the database containing the cleaned file

            Returns:
                    X (object) : Input dataframe containing messages
                    Y (object) : All output categories
                    target_names (list) : Names of all categories
    '''
    
    engine = create_engine('sqlite:///%s' % database_filepath)
    df = pd.read_sql('SELECT * FROM Disaster_response_table', engine)
    X = df.iloc[:,1]
    Y = df.iloc[:,4:]
    target_names = Y.columns

    return X, Y, target_names


def tokenize(text):
    
    '''
    This tokenizing function breaks down the string messages.
    1. All text is made lower
    2. All punctuation is removed
    3. All stop words are removed
    4. Text is lemmatized
    Parameters:
                    text (str): String messages to be tokenized

            Returns:
                    tokenized_and_stopwords_lemm (list) : tokenized list of words
    '''
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized = tokenizer.tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokenized_and_stopwords = [words for words in tokenized if not words in stop_words] 
    tokenized_and_stopwords_lemm = [WordNetLemmatizer().lemmatize(w) for w in tokenized_and_stopwords]
    return tokenized_and_stopwords_lemm


def build_model():
    
    '''
    This function initializes a model and builds a pipeline
    The function uses a Randomforest Multioutput classifier
    It uses TFidVectorizer for feature creation from the tokenized text
    Finally GridsearchCV is used to perform multiple iterations with different
    parameters to find the best model.
    Parameters:
                   
            Returns:
                    pipeline (object) : Pipeline object for training 
    '''
    rf = RandomForestClassifier()
    classifier = MultiOutputClassifier(rf)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', classifier)
    ])
    parameters = {
    'clf__estimator__bootstrap': (True, False),
     'clf__estimator__n_estimators': [50, 30],
    }
    cv = GridSearchCV(pipeline , param_grid = parameters , n_jobs=-1 , verbose=3 , scoring = 'f1_macro')
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    This function provides the precision , recall and f1 scores
    on the test data of the trained model
    Parameters:
                   model (object) : trained model
                   X_test (object) : test set
                   Y_test (object) : target values
                   category_names (list) : category names
            Returns:
                    prints the mean f1 , precision and recall scores 
    '''
    y_pred = model.predict(X_test)
    

    
    return print(classification_report(Y_test.values, y_pred , target_names = category_names ))


def save_model(model, model_filepath):
    '''
    Saves the trained model
    Parameters:
                   model (object) : trained model
                   model_filepath (str) : path to store model

            Returns:
                    
    '''
    joblib.dump(model, model_filepath)
    pass


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