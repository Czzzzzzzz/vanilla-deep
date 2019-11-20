import pandas as pd

from gensim.models import KeyedVectors
from gensim.models import Word2Vec

# command line to convert glove file to word2vec file.
# python -m gensim.scripts.glove2word2vec --input <glove_file> --output <w2v_file>

def read_glove_embedding(fn="../../data/embedding/glove.6B/w2v.6B.100d.txt"):
    wv = KeyedVectors.load_word2vec_format(fn, binary=False)
    return wv

def read_sentence_embedding(fn="../../data/embedding/medical_sieve_training_set1__cbow.txt"):
    wv = pd.read_csv(fn, sep=" ", header=-1)
    wv = wv.iloc[:, :-1]
    return wv

