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
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report


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


def tokenize(text):
    pass


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


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
