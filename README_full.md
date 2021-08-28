# Disaster Response ML Workflow Project

The purpose of this project is to build a model for an API that classifies disaster messages. The idea is to create a machine learning pipeline to categorize these events so that these messages can be send to an appropriate disaster relief agency. Project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data. This project is built using Flask, Bootstrap, Plotly, NLTK to work with human language data and Pandas for data wrangling. Data is provided by [Figure Eight](https://www.figure-eight.com/).

[<img src="img/app_main_800x366.png" width="800" height="366" alt="app main">](https://dsdn-cl-webapp-test.herokuapp.com/)

[<img src="img/app_classify_800x392.png" width="800" height="392" alt="app classification">](https://dsdn-cl-webapp-test.herokuapp.com/)

### Presentation

A live version of the dashboard is hosted on Heroku [**here**](https://dsdn-cl-webapp-test.herokuapp.com/).

## Environment

The development environment used for working with the project was created on a Windows 10 pc with [devcontainers](https://code.visualstudio.com/docs/remote/containers).

The environment in the devcontainer is:

- Python 3.9.6

The main libraries used for this project are ([`requirements.txt`](requirements.txt) includes dependencies):

- flask
- pandas
- plotly
- gunicorn

## Files

- `.devcontainer` directory is for building the local development environment, check [here](https://code.visualstudio.com/docs/remote/devcontainerjson-reference) for more info.
- `.vscode` directory is for setting up [Visual Studio Code](https://code.visualstudio.com/) and debug configurations.
- `worldbankapp` is the flask application, including templates and routes.
- `scripts` is the where the data acquisition, processing and analysis occurs.
- `Dockerfile.dev` is the Dockerfile used to build the development container.
- `Procfile` is used to deploy the web app on [Heroku](http://heroku.com/).
- `requirements.txt` contains the required python libraries to run the app.
- `worldbank.py` imports the app and can be used to run the app on a local server.

## Deployment

The web app is [**deployed on Heroku**](https://dsdn-cl-webapp-test.herokuapp.com/), for demonstration purposes.

## Running the app locally

To run locally you can clone this repository in a Windows 10 pc and use VS Code to open in a remote container. Please check instructions [here](https://code.visualstudio.com/docs/remote/containers-tutorial) if you are not familiar with devcontainers.
If you are developing a Linux machine, it may be easier to create a [virtual environment](https://docs.python.org/3/tutorial/venv.html) and run the app there.

## Acknowledgment

This project has been created as part of the [Udacity Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) course.
