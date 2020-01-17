import numpy as np
import pandas as pd
import nltk
import io
from nltk import word_tokenize
from string import punctuation


nltk.download('punkt')


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    vocab_size, dim = map(int, fin.readline().split())
    word_to_vec_map = {}

    for line in fin:
        tokens = line.rstrip().split(' ')
        word_to_vec_map[tokens[0]] = np.array(tokens[1:], dtype=np.float64)
    
    return word_to_vec_map


def word_to_vec(word, word_to_vec_map):
    if word not in word_to_vec_map:
        word = "unknown"  # for words that don't exist

    return word_to_vec_map[word]


def valid_token(token):
    return True


def convert_to_word_vectors(comment, word_to_vec_map):
    word_tokens = word_tokenize(comment)
    word_tokens = list(filter(valid_token, word_tokens))
    vectors = np.zeros((len(word_tokens), 300))

    for i, word in enumerate(word_tokens):
        vectors[i] = word_to_vec(word, word_to_vec_map)

    return vectors