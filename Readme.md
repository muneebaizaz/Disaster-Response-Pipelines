# Table of Contents
1. <a href="#1.-Introduction">Introduction</a>
2. <a href="#2.-Installations">Installations</a>
3. <a href="#3.-Files">Files</a>
4. <a href="#3.-How-to-run-the-files">How to run the files</a>
5. <a href="#4.-Links-and-Acknowledgements">Links and Acknowledgements</a>


## 1. Introduction 
This repository contains consists of a project in affiliation with the Data Science nano degree with Udacity. The project analyses real disaster response messages and categorizes and classifies them to their respective departments. Therefore different types of disaster messages get forwarded to different organizations that can be responsible for handling such problems. The task tackles aspects of data engineering and data science and ETL and MLPs are created ending with a basic web app where if you enter a message it will tell you the departments or categories with a higher affiiliation with that message. 


## 2. Installations 
The following modules have been used in the work done in this Project.
1. **Pandas and Numpy:** Modules used for data manipulation.
2. **SQL Alchemy:** For creating databases and tables.
3. **NLTK:** For tokenizing and creating features out of text
4. **Sklearn:** For data modelling and Validation

## 3. Files 
The repository consists of a total of 4 Files.
1. This **Readme** file.
2. The python file **process_data**. This file loads the already provided CSV files and performs some data cleaning and stores the result in a database.
3. The python file **train_classifier**. This file trains and perfroms multiclass classification.
4. The python file **run**. This file loads the cleaned data from the database and the trained model and provides a web link where you can analyse the results.
3. A CSV file consisting of all the disaster messages  **messages.csv**.
4. A CSV file consisting of all the labeled categories for each corresponding message  **categories.csv**

## 3. How to run the files 

1. Run the following commands in the project's root directory to set up your database and model.
    * To run ETL pipeline that cleans data and stores in database "**python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db**"
    * To run ML pipeline that trains classifier and saves "**python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl**"
2. Run the following command in the app's directory to run your web app."**python run.py**"
3. Go to http://0.0.0.0:3001/    


## 5. Links and Acknowledgements 
1. [Udacity](https://www.udacity.com/): This project is part of the data science nano degree I am doing in collaboration with Udacity.
2. [Figure Eight](https://appen.com/figure-eight-is-now-appen/): The company providing the pre-labeled dataset for the analysis.



```python

```
