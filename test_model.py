import unittest

import torch

import model


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
        encoder = model.Encoder(embed, self.hidden_size, pad_value=0)
        contex, (h, c) = encoder(x)

        self.assertEqual(contex.shape,
                         (self.seq_len, self.batch_size, 2 * self.hidden_size))
        self.assertEqual(h.shape, (self.batch_size, self.hidden_size))
        self.assertEqual(c.shape, (self.batch_size, self.hidden_size))

    def test_decoder(self):
        contex = torch.randn(
            self.seq_len, self.batch_size, 2 * self.hidden_size)
        h = torch.randn(self.batch_size, self.hidden_size)
        c = torch.randn(self.batch_size, self.hidden_size)
        y = torch.randint(self.vocab_size, (self.y_seq_len, self.batch_size))

        embed = torch.nn.Embedding(self.vocab_size, self.embed_size)
        self.attention = model.Attention(
            decoder_hidden_size=2 * self.hidden_size,
            encoder_hidden_size=2 * self.hidden_size,
        )
        decoder = model.Decoder(
            embed=embed,
            input_size=self.embed_size + 2 * self.hidden_size,
            hidden_size=self.hidden_size,
            attention=self.attention,
            sos_value=1,
        )

        y_preds = decoder((h, c), contex, y)
        self.assertEqual(
            y_preds.shape, (self.y_seq_len, self.batch_size, self.vocab_size))

    def test_attention(self):
        encoder_hidden_size = 3
        decoder_hidden_size = 6
        batch_size = 4
        seq_len = 5
        encoder_output = torch.randn(seq_len, batch_size, encoder_hidden_size)
        decoder_hidden = torch.randn(batch_size, decoder_hidden_size)

        atten = model.Attention(
            decoder_hidden_size=decoder_hidden_size,
            encoder_hidden_size=encoder_hidden_size,
        )
        context = atten(decoder_hidden, encoder_output)
        self.assertEqual(context.shape, (batch_size, encoder_hidden_size))

    def test_seq2seq(self):
        x = torch.randint(self.vocab_size, (self.seq_len, self.batch_size))
        y = torch.randint(self.vocab_size, (self.y_seq_len, self.batch_size))

        seq2seq = model.Seq2Seq(
            vocab_size=self.vocab_size,
            embed_size=self.embed_size,
            hidden_size=self.hidden_size,
            pad_value=0,
            sos_value=1,
        )
        y_preds = seq2seq(x, y)
        self.assertEqual(y_preds.shape,
                         (self.y_seq_len, self.batch_size, self.vocab_size))


if __name__ == '__main__':
    unittest.main()
