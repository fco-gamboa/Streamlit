import streamlit as st
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
import pickle

st.title("Modèle Word2Vec")

#-----------------------------------------------
# Load the Model
#-----------------------------------------------

with open("word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)
with open("idx2word.pkl", "rb") as f:
    idx2word = pickle.load(f)

vocab_size = len(word2idx) + 1


embedding_dim = 300
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(GlobalAveragePooling1D())
model.add(Dense(vocab_size, activation='softmax'))

model.load_weights("word2vec.h5")

#-----------------------------------------------
# Fonctions to compute and find Similarity
#-----------------------------------------------
vectors = model.layers[0].trainable_weights[0].numpy()

import numpy as np
from sklearn.preprocessing import Normalizer

def dot_product(vec1, vec2):
    return np.sum((vec1*vec2))

def cosine_similarity(vec1, vec2):
    return dot_product(vec1, vec2)/np.sqrt(dot_product(vec1, vec1)*dot_product(vec2, vec2))

def find_closest(word_index, vectors, number_closest):
    list1=[]
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist,index])
    return np.asarray(sorted(list1,reverse=True)[:number_closest])

def compare(index_word1, index_word2, index_word3, vectors, number_closest):
    list1=[]
    query_vector = vectors[index_word1] - vectors[index_word2] + vectors[index_word3]
    normalizer = Normalizer()
    query_vector =  normalizer.fit_transform([query_vector], 'l2')
    query_vector= query_vector[0]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist,index])
    return np.asarray(sorted(list1,reverse=True)[:number_closest])

def print_closest(word, number=10):
    index_closest_words = find_closest(word2idx[word], vectors, number)
    for index_word in index_closest_words :
        print(idx2word[index_word[1]]," -- ",index_word[0])

#-----------------------------------------------
# THE APP : 
'''FUNCTIONS:
- Predict the sentiment of a phrase
-  '''
#-----------------------------------------------
st.header('Review Sentiment Prediction')
review= st.text_input('Enter a review:')

sentiment = model.predict(review)

if sentiment:
    st.write(f'Sentiment:{sentiment}')

st.header('Arithmetic ')




word= st.text_input('Enter a word:')

num_closest= st.number_input('Choose the number of similar words')
st.write(print_closest(word, num_closest)) 






