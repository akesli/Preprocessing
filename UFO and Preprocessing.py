import pandas as pd

# URL of the CSV file
url = "https://assets.datacamp.com/production/repositories/1816/datasets/a5ebfe5d2ed194f2668867603b563963af4769e9/ufo_sightings_large.csv"

# Read the CSV file directly into a DataFrame
ufo = pd.read_csv(url)

# Print the DataFrame info
print(ufo.info())

# Change the type of seconds to float
ufo["seconds"] = ufo['seconds'].astype('float')

# Change the date column to type datetime
ufo["date"] = pd.to_datetime(ufo['date'])

# Check the column types
print(ufo.info())

# Count the missing values in the length_of_time, state, and type columns, in that order
print(ufo[['length_of_time', 'state', 'type']].isna().sum())

# Drop rows where length_of_time, state, or type are missing
ufo_no_missing = ufo.dropna(subset=["length_of_time", "state", "type"])

# Print out the shape of the new dataset
print(ufo_no_missing.shape)

#Search time_string for numbers
import re

def return_minutes(time_string):

    # Search for numbers in time_string
    # Ensure time_string is a string
    time_string = str(time_string)

    # Search for numbers in time_string
    num = re.search("\d+", time_string)

    # Handle cases where no number is found
    if num is not None:
        return int(num.group(0))
    else:
        # Handle the case where no numbers are found in time_string
        # You can return a default value, raise an exception, or take other actions.
        return None  # Example: return None if no number is found
        
# Apply the extraction to the length_of_time column
ufo["minutes"] = ufo["length_of_time"].apply(return_minutes)

# Take a look at the head of both of the columns
print(ufo[['length_of_time', 'minutes']].head())

import numpy as np

# Check the variance of the seconds and minutes columns
print(ufo[['seconds', 'minutes']].var())

# Log normalize the seconds column
ufo["seconds_log"] = np.log(ufo['seconds'])

# Print out the variance of just the seconds_log column
print(ufo['seconds_log'].var())

# Use pandas to encode us values as 1 and others as 0
ufo["country_enc"] = ufo["country"].apply(lambda val:1 if val =='us' else 0)

# Print the number of unique type values
print(len(ufo['type'].unique()))

# Create a one-hot encoded set of the type values
type_set = pd.get_dummies(ufo['type'])

# Concatenate this set back to the ufo DataFrame
ufo = pd.concat([ufo, type_set], axis=1)

# Look at the first 5 rows of the date column
print(ufo['date'].head())

# Extract the month from the date column
ufo["month"] = ufo["date"].dt.month

# Extract the year from the date column
ufo["year"] = ufo["date"].dt.year

# Take a look at the head of all three columns
print(ufo[['date', 'month', 'year']].head())

# Take a look at the head of the desc field
print(ufo['desc'].head())

from sklearn.feature_extraction import TfidfVectorizer

# Instantiate the tfidf vectorizer object
vec = TfidfVectorizer()

# Fit and transform desc using vec
desc_tfidf = vec.fit_transform(ufo['desc'])

# Look at the number of columns and rows
print(desc_tfidf.shape)

# Make a list of features to drop
to_drop = ['country', 'city', 'lat', 'long', 'state', 'date', 'recorded', 'seconds', 'minutes', 'desc', 'length_of_time']

# Drop those features
ufo_dropped = ufo.drop(to_drop, axis=1)

# Let's also filter some words out of the text vector we created
filtered_words = words_to_filter(vocab, vec.vocabulary_, desc_tfidf, 4)

#Create X and y sets of the data
X = ufo[['seconds_log', 'changing', 'chevron', 'cigar', 'circle', 'cone', 'cross', 'cylinder', 'diamond', 'disk', 'egg', 'fireball', 'flash', 'formation', 'light', 'other', 'oval', 'rectangle',
       'sphere', 'teardrop', 'triangle', 'unknown', 'month', 'year']]
y = ufo[['country_enc']]
# Take a look at the features in the X set of data
print(X.columns)

# Split the X and y sets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state=42)

# Fit knn to the training sets
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Print the score of knn on the test sets
print(knn.score(X_test, y_test))

# Use the list of filtered words we created to filter the text vector
filtered_text = desc_tfidf[:, list(filtered_words)]

# Split the X and y sets using train_test_split, setting stratify=y 
X_train, X_test, y_train, y_test = train_test_split(filtered_text.toarray(), y, stratify = y, random_state=42)

from sklearn.naive_bayes import MultinomialNB
# Fit nb to the training sets
nb.fit(X_train, y_train)

# Print the score of nb on the test sets
print(nb.score(X_test, y_test))