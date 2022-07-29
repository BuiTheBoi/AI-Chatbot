# Other imports

import string
import numpy as np
import pickle
import json
import random

# Tensorflow imports
from tensorflow.keras.optimizers import SGD
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

# Gets rid of duplicates and sorts in order
words = sorted(set(words))
classes = sorted(set(classes))

# Writing binaries pickle
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# ---------------------- Creating bag of words ----------------------
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]  # Accessing the word list
    word_patterns = [lemmatizer.lemmatize(
        word.lower()) for word in word_patterns]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])  # features
train_y = list(training[:, 1])  # labels


model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y),
                 epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print('Success!')
