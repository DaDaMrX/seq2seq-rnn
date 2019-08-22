import unittest

from dataset import DataBuilder, Dataset, load_embedding


class TestDataset(unittest.TestCase):
    data_dir = 'data'
    embedding_path = 'embedding/glove.42B.300d.txt'
    vocab_size = 10000
    seq_max_len = 30

    def test_data_builder(self):
        pairs, vocab = DataBuilder.build(
            data_dir=self.data_dir,
            max_len=self.seq_max_len,
            vocab_size=self.vocab_size,
            use_cache=False,
        )
        dataset = Dataset(pairs, vocab)
        self.assertEqual(len(dataset), 137990)
        self.assertEqual(len(vocab), self.vocab_size)

    def test_load_embedding(self):
        pairs, vocab = DataBuilder.build(
            data_dir=self.data_dir,
            max_len=self.seq_max_len,
            vocab_size=self.vocab_size,
            use_cache=True,
        )
        self.assertEqual(len(vocab), self.vocab_size)
        embedding = load_embedding(
            embedding_path=self.embedding_path,
            vocab=vocab,
            use_cache=False,
        )
        self.assertEqual(embedding.shape, (self.vocab_size, 300))
 