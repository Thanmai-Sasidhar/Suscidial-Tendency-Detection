import numpy as np

def load_glove_embeddings(glove_file):
    embeddings = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Load GloVe
glove_path = "/Users/ats/Desktop/archive (2)/glove/glove.840B.300d.txt"
glove_embeddings = load_glove_embeddings(glove_path)

# Test
print(f"Vector for 'hello': {glove_embeddings.get('hello', 'Not found')}")
