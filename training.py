# Other imports

import string
import numpy as np
import pickle
import json
import random

# Tensorflow imports
# from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential

# NLTK imports
import nltk  # Natural Language Toolkit
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())  # Returns a dictionary

words = []
classes = []
documents = []
ignore_letters = [p for p in string.punctuation]

for intent in intents['intents']:
    # Patterns represent the input from users from json file
    for pattern in intent['patterns']:
        # Tokenizing (splitting) each string into individual words
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Eliminating any useless punctuation
words = [lemmatizer.lemmatize(word)
         for word in words if word not in ignore_letters]
words = sorted(set(words))  # Gets rid of duplicates and sorts in order

print(words)
