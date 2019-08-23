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
        # h = torch.cat([h[-2], h[-1]], dim=-1)
        # c = torch.cat([c[-2], c[-1]], dim=-1)
        h = h[-2] + h[-1]
        c = c[-2] + c[-1]
        return output, (h, c)


class Attention(torch.nn.Module):
    
    def __init__(self, decoder_hidden_size, encoder_hidden_size):
        super(Attention, self).__init__()
        self.linear = torch.nn.Linear(
            decoder_hidden_size, encoder_hidden_size)
        
    def forward(self, decoder_hidden, encoder_output):
        h = self.linear(decoder_hidden)
        encoder_output = encoder_output.transpose(0, 1)
        a = torch.matmul(encoder_output, h.unsqueeze(-1)).squeeze()
        a = torch.softmax(a, dim=1)
        context = (a.unsqueeze(-1) * encoder_output).sum(1)
        return context


class Decoder(torch.nn.Module):
    
    def __init__(self, embed, input_size, hidden_size, attention, sos_value):
        super(Decoder, self).__init__()
        self.sos_value = sos_value
        self.embed = embed
        self.lstm_cell = torch.nn.LSTMCell(input_size, hidden_size)
        self.attention = attention
        self.out = torch.nn.Linear(hidden_size, embed.num_embeddings)
        
    def forward(self, hidden, encoder_output, y):
        y_preds = []
        h, c = hidden
        x = torch.empty(*y.size()[1:], dtype=torch.long)
        x = x.fill_(self.sos_value)
        for i in range(len(y)):
            context = self.attention(
                torch.cat([h, c], dim=1),
                encoder_output,
            )
            x = self.embed(x)
            h, c = self.lstm_cell(
                torch.cat([x, context], dim=1),
                (h, c)
            )
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
        self.attention = Attention(
            decoder_hidden_size=2 * hidden_size,
            encoder_hidden_size=2 * hidden_size,
        )
        self.decoder = Decoder(
            embed=self.embed,
            input_size=self.embed.embedding_dim + 2 * hidden_size,
            hidden_size=hidden_size,
            attention=self.attention,
            sos_value=sos_value,
        )
        
    def forward(self, x, y):
        encoder_output, (h, c) = self.encoder(x)
        y_preds = self.decoder((h, c), encoder_output, y)
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
