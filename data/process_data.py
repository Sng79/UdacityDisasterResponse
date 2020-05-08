"""
DATA
Disaster Response Pipeline Project - Udacity
    1) CSV file containing messages (disaster_messages.csv)
    2) CSV file containing categories (disaster_categories.csv)
    3) SQLite destination database (DisasterResponse.db)
"""    

#Load libraries 
import sys
import sqlite3
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine


#Load data function
"""
    Arguments:
        messages_filepath -> path to messages csv file
        categories_filepath -> path to categories csv file
    Output:
        df -> Loaded data as pd.DataFrame
"""

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

# merge dataframes
    df = messages.merge(categories, on=['id'], how='left')

    return df

def clean_data(df):
    """
    Clean Data function
    
    Arguments:
        df -> raw data Pandas DataFrame
    Outputs:
        df -> clean data Pandas DataFrame
    """

# Splits categories in dataframe and converts them into binary
    categories = df['categories'].str.split(';', expand=True)

    #select first row
    row = categories.iloc[0]
    #get list of categories from that row , apply lambda function untill second to last character
    category_colnames = row.transform(lambda x: x[:-2]).tolist()
    #rename cols to categories
    categories.columns = category_colnames
    # Convert  category numbers
    for column in categories:
        # set every value to be the last character of a string
        categories[column] = categories[column].transform(lambda x: x[-1:])
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # Drop the  categories  from `df`
    df.drop('categories', axis = 1, inplace = True)
    
    # Concatenate   dataframe with the new `categories` 
    df = pd.concat([df, categories], axis = 1)
    # Drop duplicates
    df.drop_duplicates(inplace = True)
    # Remove rows with a  value of 2 from df
    df = df[df['related'] != 2]
    
    return df


#Save the clean dataset into an sqlite database
def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Messagescategories', engine, index=False,if_exists = 'replace')


def main():
    """Main Data Processing function implementing the pipeline:
        1) Extracts data from .csv
        2) Cleans data
        3) Loads data to SQLite database
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
