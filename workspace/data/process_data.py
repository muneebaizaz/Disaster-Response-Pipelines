import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    This function takes the filepath of the messages and 
    categories csv files and merges the data together and returns
    the merged dataframe.
    Parameters:
                    messages_filepath (str): filepath for messages csv file
                    categories_filepath (str): filepath for categories csv file

            Returns:
                    merged_df (object): merged pandas dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    merged_df = categories.merge(messages , on ='id')

    return merged_df 


def clean_data(df):
    
    '''
    This function cleans the dataframe by performing multiple tasks.
    1. Values in the category column are split on ";" character so 
    each value becomes a seperate column.
    2. Columns are renamed using the values from the first row.
    Parameters:
                    df (object): Dataframe object

            Returns:
                    df (object) : cleaned dataframe 
    '''
    
    categories = df.iloc[:,1].str.split(";" , expand=True)
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x : x[0:-2])
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x : x[-1])

        # convert column from string to numeric
        categories[column] = categories[column].apply(lambda x : int(x))
    # few labels of 2 exist they are replaced with 1
    categories.related.replace(2,1,inplace=True)    
    df = df.drop(columns = 'categories')
    df = pd.concat((df,categories) , axis = 1)
    df.drop_duplicates(inplace=True)



    return df 


def save_data(df, database_filename):
    '''
    This function saves the cleaned dataframe and creates a database by
    taking the database filename as input and stores the dataframe as a table.
    
    Parameters:
                    df (object): dataframe object
                    database_filename (str): name of the created database
    
    '''
    
    engine = create_engine('sqlite:///%s' % database_filename)
    df.to_sql('Disaster_response_table', engine, index=False , if_exists = 'replace')
    pass  


def main():
    
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