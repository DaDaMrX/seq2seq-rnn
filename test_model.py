import unittest

import torch

from model import Encoder, Decoder, Seq2Seq


class TestModel(unittest.TestCase):

    vocab_size = 10
    embed_size = 5
    batch_size = 4
    hidden_size = 4
    seq_len = 5
    y_seq_len = 4

    def test_encoder(self):
        x = torch.randint(self.vocab_size, (self.seq_len, self.batch_size))

        embed = torch.nn.Embedding(self.vocab_size, self.embed_size)
        encoder = Encoder(embed, self.hidden_size, pad_value=0)
        contex, (h, c) = encoder(x)

        contex_shape = (self.seq_len, self.batch_size, 2 * self.hidden_size)
        h_shape = (self.batch_size, 2 * self.hidden_size)
        c_shape = (self.batch_size, 2 * self.hidden_size)

        self.assertEqual(contex.shape, contex_shape)
        self.assertEqual(h.shape, h_shape)
        self.assertEqual(c.shape, c_shape)

    def test_decoder(self):
        contex = torch.randn(self.seq_len, self.batch_size,
                             2 * self.hidden_size)
        h = torch.randn(self.batch_size, 2 * self.hidden_size)
        c = torch.randn(self.batch_size, 2 * self.hidden_size)
        y = torch.randint(self.vocab_size, (self.y_seq_len, self.batch_size))

        embed = torch.nn.Embedding(self.vocab_size, self.embed_size)
        decoder = Decoder(embed, 2 * self.hidden_size, sos_value=1)
        y_preds = decoder((h, c), contex, y)

        y_preds_shape = (self.y_seq_len, self.batch_size, self.vocab_size)

        self.assertEqual(y_preds.shape, y_preds_shape)

    def test_seq2seq(self):
        x = torch.randint(self.vocab_size, (self.seq_len, self.batch_size))
        y = torch.randint(self.vocab_size, (self.y_seq_len, self.batch_size))

        seq2seq = Seq2Seq(
            vocab_size=self.vocab_size,
            embed_size=self.embed_size,
            hidden_size=self.hidden_size,
            pad_value=0,
            sos_value=1,
        )
        y_preds = seq2seq(x, y)

        y_preds_shape = (self.y_seq_len, self.batch_size, self.vocab_size)

        self.assertEqual(y_preds.shape, y_preds_shape)
