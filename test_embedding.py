import unittest

from dataset import DataBuilder, Vocab
from embedding import load_embedding


class TestEmbedding(unittest.TestCase):

    def test_load(self):
        vocab_size = 10000
        pairs, vocab = DataBuilder.build(
            data_dir='data',
            min_len=30,
            vocab_size=vocab_size,
        )
        embedding_path = 'embedding/glove.twitter.27B.200d.txt'
        embedding = load_embedding(embedding_path, vocab)
        self.assertEqual(embedding.shape, (vocab_size, 200))
