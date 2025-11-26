import streamlit as st
import numpy as np
from sklearn.preprocessing import Normalizer
import keras 
from keras import Sequential
#from tensorflow.keras.models import load_model
from keras.layers import Embedding, Dense, GlobalAveragePooling1D
import pickle

st.title("🔤 Word2Vec Explorer")

#-----------------------------------------------
# Load ONLY ONCE the tokenizer maps (word2idx, idx2word) and the Model
#-----------------------------------------------
@st.cache_resource
def get_model():
    vocab_size = 72541
    embedding_dim = 300
    model = Sequential([
        Embedding(vocab_size, embedding_dim),
        GlobalAveragePooling1D(),
        Dense(vocab_size, activation='softmax')
    ])
    model = keras.models.load_model("word2vec_cbow.keras")
    #model.load_weights("word2vec_cbow.h5")
    return model

@st.cache_resource
def load_mappings():
    with open("word2idx.pkl", "rb") as f:
        word2idx = pickle.load(f)
    with open("idx2word.pkl", "rb") as f:
        idx2word = pickle.load(f)
    return word2idx, idx2word

model = get_model()
word2idx, idx2word = load_mappings()
vectors = model.layers[0].get_weights()[0]

#-----------------------------------------------
# Similarity functions
#-----------------------------------------------
def dot_product(vec1, vec2):
    return np.sum(vec1 * vec2)

def cosine_similarity(vec1, vec2):
    return dot_product(vec1, vec2) / np.sqrt(dot_product(vec1, vec1) * dot_product(vec2, vec2))

def find_closest(word_index, vectors, number_closest):
    results = []
    query_vector = vectors[word_index]

    for index, vector in enumerate(vectors):
        if index != word_index:
            dist = cosine_similarity(query_vector, vector)
            results.append((dist, index))

    results = sorted(results, reverse=True)[:number_closest]
    return results

def compare(idx1, idx2, idx3, vectors, number_closest):
    normalizer = Normalizer()
    query_vec = vectors[idx1] - vectors[idx2] + vectors[idx3]
    query_vec = normalizer.fit_transform([query_vec])[0]

    results = []
    for index, vector in enumerate(vectors):
        dist = cosine_similarity(query_vec, vector)
        results.append((dist, index))

    results = sorted(results, reverse=True)[:number_closest]
    return results


# ================================================
#            🔍 Similar Words Explorer
# ================================================
st.header("🔎 Trouver les mots similaires")

word = st.selectbox("Choisissez un mot :", list(word2idx.keys()))
num_similar = st.slider("Nombre de mots similaires :", 5, 30, 10)

if st.button("Afficher les mots similaires"):
    st.subheader(f"Mots les plus proches de : **{word}**")
    closest_words = find_closest(word2idx[word], vectors, num_similar)

    for dist, idx in closest_words:
        st.write(f"**{idx2word[idx]}** — similarity: `{dist:.4f}`")


# ================================================
#            🧮 Semantic Word Arithmetic
# ================================================
st.header("🧠 Calcul sémantique : a - b + c = ?")

col1, col2, col3 = st.columns(3)

with col1:
    word_a = st.selectbox("a", list(word2idx.keys()), key="a")

with col2:
    word_b = st.selectbox("b", list(word2idx.keys()), key="b")

with col3:
    word_c = st.selectbox("c", list(word2idx.keys()), key="c")

num_results = st.slider("Nombre de résultats :", 3, 20, 5)

if st.button("Calculer analogie"):
    st.subheader(f"Résultat de : **{word_a} - {word_b} + {word_c}**")
    results = compare(
        word2idx[word_a], 
        word2idx[word_b], 
        word2idx[word_c],
        vectors,
        num_results
    )

    for dist, idx in results:
        st.write(f"**{idx2word[idx]}** — similarity: `{dist:.4f}`")
