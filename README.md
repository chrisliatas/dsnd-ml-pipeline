# Disaster Response ML Workflow Project

The purpose of this project is to build a model for an API that classifies disaster messages. The idea is to create a machine learning pipeline to categorize these events so that these messages can be send to an appropriate disaster relief agency. Project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data. This project is built using Flask, Bootstrap, Plotly, NLTK to work with human language data and Pandas for data wrangling. Data is provided by [Figure Eight](https://www.figure-eight.com/).

[<img src="img/app_main_800x366.png" width="800" height="366" alt="app main">](https://dsnd-disaster-resp.herokuapp.com/)

[<img src="img/app_classify_800x392.png" width="800" height="392" alt="app classification">](https://dsnd-disaster-resp.herokuapp.com/)

### Presentation

A live version of the dashboard is hosted on Heroku [**here**](https://dsnd-disaster-resp.herokuapp.com/).

## Environment

The development environment used for working with the project was created on a Windows 10 pc with [devcontainers](https://code.visualstudio.com/docs/remote/containers).

The environment in the container is:

- Python 3.9.6

The main libraries used for this project are ([`requirements.txt`](requirements.txt) includes dependencies):

- jupyter
- scikit-learn
- nltk
- sqlalchemy
- flask
- pandas
- plotly
- gunicorn
- seaborn

## Files and project file structure

Important files:

- `app` is the flask application, including templates, static files, and routes.
- `data` is the where the data and produced database are kept.
- `data/process_data.py` is the python script is the ETL pipeline, where data processing, analysis and database creation occurs.
- `data/disaster_messages.csv` the dataset used for analysis containing more than 26000 messages, provided by Figure Eight.
- `data/disaster_categories.csv` categories for the above messages' classification, provided by Figure Eight.
- `models/train_classifier.py` is the ML workflow script, including data loading, building a text processing and machine learning pipeline, train and tune a model using [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), results output for the test set and Exporting the final model as a pickle file.
- `models/classifier.pkl` this is the saved trained model. It is a large file added to git using [`git lfs`](https://git-lfs.github.com/)
- `.devcontainer` directory is for building the local development environment, check [here](https://code.visualstudio.com/docs/remote/devcontainerjson-reference) for more info.
- `.vscode` directory is for setting up [Visual Studio Code](https://code.visualstudio.com/) and debug configurations.
- `.github/workflows` github [actions](https://github.com/features/actions) for CI/CD workflows.
- `Dockerfile.dev` is the Dockerfile used to build the development container.
- `Procfile` is used to deploy the web app on [Heroku](http://heroku.com/).
- `requirements.txt` contains the required python libraries to run the app.
- `run.py` imports the app and can be used to run the app on a local server.

Directories and files structure:

```
- app
|- static
| |- css
| | |- style.css  # custom style sheet used for the app template
| - templates
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process
|- disaster_messages.csv  # data to process
|- ETL Pipeline Preparation.ipynb  # jupyter file for initial data exploration and development.
|- process_data.py
|- DisasterResponse.db   # database to save clean data to

- img  # contains app screenshots for this README
|- app_classify_800x392.png
|- app_main_800x366.png

- models
|- train_classifier.py
|- classifier.pkl  # saved model
|- ML Pipeline Preparation.ipynb  # jupyter file for initial development and ML pipeline attempts.

- LICENSE
- README.md
- requirements.txt
```

## Deployment

The web app is [**deployed on Heroku**](https://dsdn-cl-webapp-test.herokuapp.com/), for demonstration purposes.

## Running the app locally

To run locally you can clone this repository in a Windows 10 pc and use VS Code to open in a remote container. Please check instructions [here](https://code.visualstudio.com/docs/remote/containers-tutorial) if you are not familiar with devcontainers.
If you are developing a Linux machine, it may be easier to create a [virtual environment](https://docs.python.org/3/tutorial/venv.html) and run the app there.

### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

   - To run ETL pipeline that cleans data and stores in database
     `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
   - To run ML pipeline that trains classifier and saves
     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
   `python run.py`

3. Go to http://0.0.0.0:3001/

## Acknowledgments

- This project has been created as part of the [Udacity Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) course.
- Data is provided by [Figure Eight](https://www.figure-eight.com/)
