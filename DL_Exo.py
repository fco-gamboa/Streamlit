import pandas as pd
import numpy as np
import re
import unicodedata
import nltk
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
from tqdm import tqdm

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------
df = pd.read_csv("MovieReview.csv")
print(df.head())
print(df.shape)

df = df.drop('sentiment', axis=1)

# ------------------------------------------------
# NLTK
# ------------------------------------------------
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

stop_words = stopwords.words('english')

# ------------------------------------------------
# FUNCTIONS
# ------------------------------------------------
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r"\s+", " ", w)
    w = re.sub(r"[^a-zA-Z?.!]+", " ", w)
    w = re.sub(r'\b\w{0,2}\b', '', w)

    mots = word_tokenize(w.strip())
    mots = [mot for mot in mots if mot not in stop_words]
    return " ".join(mots).strip()

# Apply preprocessing
tqdm.pandas()
df.review = df.review.progress_apply(preprocess_sentence)

# ------------------------------------------------
# TOKENIZER
# ------------------------------------------------
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(df.review)

word2idx = tokenizer.word_index
idx2word = tokenizer.index_word
vocab_size = len(word2idx) + 1

# Save them for streamlit app
import pickle
with open("word2idx.pkl", "wb") as f:
    pickle.dump(word2idx, f)
with open("idx2word.pkl", "wb") as f:
    pickle.dump(idx2word, f)

print("Vocab size:", vocab_size)

# ------------------------------------------------
# BUILD TRAINING DATA (CBOW)
# ------------------------------------------------

def generate_cbow(tokens, window_size):
    """
    Returns:
    X: list of context windows (length = window_size*2)
    y: list of center words
    """
    X = []
    y = []

    for i in range(window_size, len(tokens) - window_size):
        context = []
        for j in range(i - window_size, i + window_size + 1):
            if j != i:
                context.append(tokens[j])
        X.append(context)       # context words
        y.append(tokens[i])     # center word

    return X, y

WINDOW_SIZE = 2

X, y = [], []

for review in df.review:
    sentences = review.split(".")
    for s in sentences:
        tokens = tokenizer.texts_to_sequences([s])[0]
        if len(tokens) > WINDOW_SIZE * 2:
            Xi, yi = generate_cbow(tokens, WINDOW_SIZE)
            X.extend(Xi)
            y.extend(yi)

X = np.array(X)
y = np.array(y).reshape(-1, 1)

print("X shape:", X.shape)
print("y shape:", y.shape)

# ------------------------------------------------
# MODEL (CBOW)
# ------------------------------------------------
embedding_dim = 100

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=WINDOW_SIZE * 2),
    GlobalAveragePooling1D(),
    Dense(vocab_size, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# ------------------------------------------------
# TRAIN
# ------------------------------------------------
model.fit(X, y, batch_size=128, epochs=10)

# ------------------------------------------------
# SAVE
# ------------------------------------------------
model.save("word2vec_cbow.keras") # not h5 for the full model in a Mac ( h5 is depricated)
print("Model saved!")




