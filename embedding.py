import os
import pickle

import numpy as np
import pandas as pd

from dataset import DataBuilder, Vocab


def load_embedding(embedding_path, vocab):
    pickle_path = os.path.join(
        os.path.split(embedding_path)[0], 'embedding.pickle')
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            embedding = pickle.load(f)
        print(f'Loaded from {pickle_path}.')
        return embedding

    print('Loading embedding file...')
    df = pd.read_csv(
        embedding_path, sep=' ', index_col=0, header=None, quoting=3)
    print('Load embedding done.')
    vocab_size = len(vocab)
    embed_size = df.shape[1]
    embedding = np.empty((vocab_size, embed_size), dtype=np.float32)
    print('embedding.dtype', embedding.dtype)
    for i in range(vocab_size):
        if vocab[i] in df.index:
            embedding[i] = df.iloc[i]
        else:
            embedding[i] = np.random.randn(embed_size)
    
    with open(pickle_path, 'wb') as f:
        pickle.dump(embedding, f)
        print(f'Save data to {pickle_path}.')
    return embedding


if __name__ == '__main__':
    vocab_size = 10000
    pairs, vocab = DataBuilder.build(
        data_dir='data',
        min_len=30,
        vocab_size=vocab_size,
    )
    embedding_path = 'embedding/glove.twitter.27B.200d.txt'
    embedding = load_embedding(embedding_path, vocab)

