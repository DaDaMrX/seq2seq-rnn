import unittest

from dataset import DataBuilder, Dataset


class TestDataset(unittest.TestCase):

    def test_data_builder(self):
        vocab_size = 10000
        pairs, vocab = DataBuilder.build(
            data_dir='data',
            max_len=30,
            vocab_size=vocab_size,
        )
        dataset = Dataset(pairs, vocab)
        # print('len(dataset):', len(dataset))
        self.assertEqual(len(dataset), 137952)
        self.assertEqual(len(vocab), vocab_size)    
 