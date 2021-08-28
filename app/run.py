# -*- coding: utf-8 -*-
"""
Flask app entry point. Loads data from database file and a trained model
from a .pkl file. Creates routes, builds Plotly graphs.
"""
# import libraries
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine("sqlite:///../data/DisasterResponse.db")
df = pd.read_sql_table("DisasterResponse", engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby("genre").count()["message"]
    genre_names = list(genre_counts.index)
    categories_perc = df.iloc[:, 4:].sum()
    category_names = [c.replace("_", " ") for c in df.iloc[:, 4:].columns]
    df["message_length"] = df["message"].str.len()

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            "data": [Bar(x=category_names, y=categories_perc)],
            "layout": {
                "title": "Messages categories",
                "yaxis": {"title": "Count"},
                "xaxis": {
                    "title": "Categories",
                    "tickangle": 25,
                    "title_standoff": 100,
                },
            },
        },
        {
            "data": [
                Histogram(x=df["message_length"], y=df["message_length"].value_counts())
            ],
            "layout": {
                "title": "Message length frequency",
                "yaxis": {"title": "Frequency"},
                "xaxis": {
                    "title": "Message length (chars)",
                    # "tickangle": 45,
                    "title_standoff": 100,
                },
            },
        },
        {
            "data": [Bar(x=genre_names, y=genre_counts)],
            "layout": {
                "title": "Distribution of Message Genres",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Genre"},
            },
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route("/go")
def go():
    # save user input in query
    query = request.args.get("query", "")

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        "go.html", query=query, classification_result=classification_results
    )


def main():
    app.run(host="0.0.0.0", port=3001)


if __name__ == "__main__":
    main()
