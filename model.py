import torch


class Encoder(torch.nn.Module):
    
    def __init__(self, embed, hidden_size, pad_value):
        super(Encoder, self).__init__()
        self.pad_value = pad_value
        self.embed = embed
        self.lstm = torch.nn.LSTM(
            input_size=embed.embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
        )
        
    def forward(self, x):
        lengths = (x != self.pad_value).sum(dim=0)
        x = self.embed(x)
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths, enforce_sorted=False)
        output, (h, c) = self.lstm(x)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            output, padding_value=self.pad_value)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        c = torch.cat([c[-2], c[-1]], dim=-1)
        return output, (h, c)


class Decoder(torch.nn.Module):
    
    def __init__(self, embed, hidden_size, sos_value):
        super(Decoder, self).__init__()
        self.sos_value = sos_value
        self.embed = embed
        self.lstm_cell = torch.nn.LSTMCell(
            embed.embedding_dim, hidden_size)
        self.out = torch.nn.Linear(hidden_size, embed.num_embeddings)
        
    def forward(self, hidden, contex, y):
        y_preds = []
        h, c = hidden
        x = torch.empty(*y.size()[1:], dtype=torch.long)
        x = x.fill_(self.sos_value)
        for i in range(len(y)):
            x = self.embed(x)
            h, c = self.lstm_cell(x, (h, c))
            y_pred = self.out(c)
            y_preds.append(y_pred)
            x = y[i]
        return torch.stack(y_preds)


class Seq2Seq(torch.nn.Module):
    
    def __init__(self, hidden_size, pad_value, sos_value,
                 vocab_size=None, embed_size=None, embedding=None):
        super(Seq2Seq, self).__init__()
        if embedding is not None:
            self.embed = torch.nn.Embedding.from_pretrained(
                torch.tensor(embedding), freeze=False)
        else:
            self.embed = torch.nn.Embedding(vocab_size, embed_size)
        self.encoder = Encoder(self.embed, hidden_size, pad_value)
        self.decoder = Decoder(self.embed, 2 * hidden_size, sos_value)
        
    def forward(self, x, y):
        contex, (h, c) = self.encoder(x)
        y_preds = self.decoder((h, c), contex, y)
        return y_preds


if __name__ == '__main__':
    vocab_size = 10
    embed_size = 5
    batch_size = 4
    hidden_size = 4
    seq_len = 5

    embed = torch.nn.Embedding(vocab_size, embed_size)
    encoder = Encoder(embed, hidden_size, pad_value=0)
    x = torch.randint(vocab_size, (seq_len, batch_size))
    contex, (h, c) = encoder(x)
    print(contex.shape, h.shape, c.shape)
