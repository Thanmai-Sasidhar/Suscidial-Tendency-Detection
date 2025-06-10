import numpy as np
import pickle

def load_glove_embeddings(glove_file):
    """Load GloVe embeddings from a .txt file and return a dictionary."""
    embeddings = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]  # First token is the word
            try:
                vector = np.asarray(values[1:], dtype='float32')  # Convert rest to float
                embeddings[word] = vector
            except ValueError:
                print(f"Skipping line: {line[:50]}...")
    return embeddings

# Paths
glove_txt_path = "/Users/ats/Desktop/archive (2)/glove/glove.840B.300d.txt"
glove_pkl_path = "/Users/ats/Desktop/archive (2)/glove/glove.840B.300d.pkl"

# Convert & Save as Pickle
glove_embeddings = load_glove_embeddings(glove_txt_path)

with open(glove_pkl_path, 'wb') as pkl_file:
    pickle.dump(glove_embeddings, pkl_file)

print(f"✅ GloVe embeddings saved to {glove_pkl_path}")
