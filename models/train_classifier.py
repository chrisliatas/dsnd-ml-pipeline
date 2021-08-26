# -*- coding: utf-8 -*-

"""
Script to run ML pipeline that trains a classifier, evaluates and saves the model
"""

# import libraries
import sys
import string
import re
from os import path
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk

nltk.download(["punkt", "wordnet", "stopwords"])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC

# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.externals import joblib


def load_data(database_filepath):
    """
    Load the clean dataset from an sqlite database into a pandas.Dataframe.

    Args:
        database_filepath (str): The filename to load the database file from.

    Returns:
        X (pd.DataSeries): A vector holding the messsages to consider when predicting the response.
        Y (pd.DataFrame): the corresponding response categories matrix.
        Y.columns: A list with response categories names.
    """
    engine = create_engine("sqlite:///" + database_filepath)
    # Use the database filepath without the `.db` extension for table name.
    tbl_name = path.basename(database_filepath)[:-3]
    # read the sql database with pandas `read_sql` method.
    df = pd.read_sql(tbl_name, engine)
    X = df.message
    Y = df.iloc[:, 4:]

    return X, Y, Y.columns


def decontracted(phrase, tokens=True):
    """
    Replace contractions for most unambiguous cases.
    Replace apostrophe/short words.
    For example 's (a contraction of “is”)
    Ref: https://stackoverflow.com/a/47091370/10074873

    Arguments:
        phrase (str): The string to replace contractions from.
        tokens (bool): If passing tokens or whole phrases.

    Returns:
        phrase (str): The string without contractions.
    """
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    return phrase.strip() if tokens else phrase


def tokenize(text):
    # Regular expression to detect a url within text
    url_regex = (
        "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )

    # get list of all urls using regex\
    detected_urls = re.findall(url_regex, text)

    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # tokenize text and change all capitals to lower case
    tokens = word_tokenize(text.lower())

    # Remove stop words and replace contractions for most unambiguous cases missed from stopwords.
    stop_words = set(stopwords.words("english"))
    tokens = [decontracted(tok) for tok in tokens if tok not in stop_words]

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Create clean tokens list iterating through each token
    clean_tokens = [lemmatizer.lemmatize(tok).strip() for tok in tokens]

    # Remove punctuation
    clean_tokens = list(filter(lambda tok: tok not in string.punctuation, clean_tokens))

    return clean_tokens


def build_model():
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            ("clf", LinearSVC(random_state=42, tol=1e-5)),
        ]
    )

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    """
    Save the scikit-learn created model usign Python's pickle serialization.

    Arguments:
        model (Pipeline): The string to replace contractions from.
        model_filepath (str): The filepath to save the pipeline model to.
    """
    # Since using scikit-learn, it may be better to use joblib’s replacement
    # of pickle (dump & load), which is more efficient.
    # Ref: https://scikit-learn.org/stable/modules/model_persistence.html
    # Ref: https://stackoverflow.com/a/61920454/10074873

    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
