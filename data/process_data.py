# -*- coding: utf-8 -*-

"""
Script to run ETL pipeline that cleans data and stores it in database
"""

# import libraries
import sys
from os import path
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Read disaster related messages and categories from csv files and load them into a dataframe

    Args:
        messages_filepath (str): The filepath with the file containing disaster messages.
        categories_filepath (str): The filepath with the file containing disaster categories.

    Returns:
        A merged dataframe of these data (pd.DataFrame).
    """
    try:
        # Load `data/disaster_messages.csv` into a dataframe
        messages = pd.read_csv(messages_filepath)
        # Load `data/disaster_categories.csv` into a dataframe
        categories = pd.read_csv(categories_filepath)
    except Exception as e:
        print(f"Data could not be read from given filepaths: {e}")
        exit()

    # Merge the messages and categories datasets using the common `id` column
    # Return combined dataframe
    return messages.merge(categories, on="id")


def clean_data(df):
    """
    Transform the data. Clean the dataframe created with `load_data` function.
    Extract categories from text.
    Create dummy variables from category columns.
    Remove duplicates

    Args:
        df (pd.DataFrame): The dataframe to be cleaned.

    Returns:
        A cleaned dataframe (pd.DataFrame).
    """
    # == Extract category names ==
    # Create a dataframe with the 36 individual category columns
    categories = df["categories"].str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # Use this row to extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x[:-2])
    # Rename the columns of `categories` using the new column names
    categories.columns = category_colnames

    # == Dummy variables ==
    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.slice(start=-1)
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # Drop the original categories column from `df`
    df.drop(["categories"], inplace=True, axis=1)
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # Column `related` has three categories 0, 1, 2.
    # 0 is un-related, 1 is related, 2 could be related but info is missing.
    # We will drop rows with related values == 2 since they are less than 1%
    # of the total messages and cannot be treated as the rest of the messages.
    df.drop(df[df["related"] == 2].index, inplace=True)

    # == Remove duplicates ==
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Save the clean dataset into an sqlite database.

    Args:
        df (pd.DataFrame): The dataframe to be saved in the database.
        database_filename (str): The filename to save the database file as.
    """
    engine = create_engine("sqlite:///" + database_filename)
    # Use the database filename without the `.db` extension for table name.
    tbl_name = path.basename(database_filename)[:-3]
    # save to sql database with pandas `to_sql` method.
    df.to_sql(tbl_name, engine, if_exists="replace", index=False)


def main():
    """
    The main function to run the ETL pipeline process. Get user input arguments
    Load messages from .csv files, transform the data and save the clean data
    into a database.
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()
