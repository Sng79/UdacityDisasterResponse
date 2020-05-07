# Disaster Response Pipeline Project

This project aims to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

The data set contains real messages that were sent during disaster events. A machine learning pipeline was created to categorize disaster events so that the messeages can be sent to an appropriate disaster relief agency.

The project includes web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data. 

Project Components

1. ETL Pipeline
The process_data.py script to clean data (and ETL notebook):
- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

2. ML Pipeline
The train_classifier.py (and ML Pipeline notebook) that writes a machine learning pipeline that:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

Dependencies
Python 3.5+ (I used Python 3.7)
Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
Natural Language Process Libraries: NLTK
SQLlite Database Libraqries: SQLalchemy
Web App and Data Visualization: Flask, Plotly

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
If running in Local Machine, go to  http://localhost:3001 and the app will now run

Author
Stefani Novoa 
