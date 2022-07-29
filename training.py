import random
import json
import pickle
import numpy as np

import nltk  # Natural Language Toolkit
# nltk.download('punkt')
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD


lemmatizer = WordNetLemmatizer

intents = json.loads(open('intents.json').read())  # Returns a dictionary

words = []
classes = []
documents = []
ignore_lettesr = ['?', '!', '.', ',']

for intent in intents['intents']:
    # Patterns represent the input from users from json file
    for pattern in intent['patterns']:
        # Tokenizing (splitting) each string into individual words
        word_list = nltk.word_tokenize(pattern)
        words.append(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print(documents)
