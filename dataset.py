import os
import pickle
import re

import contractions
import nltk
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from tqdm import tqdm


class DataBuilder:

    @classmethod
    def build(cls, data_dir, max_len, vocab_size, use_cache=True):
        pickle_path = os.path.join(data_dir, 'data.pickle')
        if use_cache and os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            print(f'Loaded from {pickle_path}.')
            return data
        
        convs = cls.load(data_dir)
        convs = cls.clean(convs)
        pairs = cls.make_pairs(convs)
        pairs = cls.filter(pairs, max_len)
        vocab = cls.build_vocab(pairs, vocab_size)

        with open(pickle_path, 'wb') as f:
            pickle.dump((pairs, vocab), f)
        print(f'Save data to {pickle_path}.')
        return pairs, vocab

    @classmethod
    def load(cls, data_dir):
        lines_df = pd.read_csv(
            os.path.join(data_dir, 'movie_lines.txt'),
            sep='\s*\+\+\+\$\+\+\+\s*', 
            engine='python',
            header=None,
            index_col=0,
        )
        lines_df.iloc[:, -1].fillna('', inplace=True)
        convs_df = pd.read_csv(
            os.path.join(data_dir, 'movie_conversations.txt'),
            sep='\s*\+\+\+\$\+\+\+\s*', 
            engine='python',
            header=None,
        )
        convs = []
        for line_ids in tqdm(convs_df.iloc[:, -1], desc=' Load'):
            conv = []
            for line_id in eval(line_ids):
                conv.append(lines_df.loc[line_id].iloc[-1])
            convs.append(conv)
        return convs

    @classmethod
    def clean(cls, convs):
        def _clean(sent):
            clean_dict = {
                "'bout": "about",
                "'til": "until",
                " g d": " god",
                "now 's": "now is",
                "y'know": "you know",
                "prob'ly": "probably",
                "here 's": "here is",
                "right 's": "right is",
                "ground 's": "ground is",
                "toilet 's": "toilet is",
                "door 's": "door is",
                "nothing 's": "nothing is",
                "why'd": "why did",
                "cannot": "can not",
                "that ' s": "that is",
                "i ' m": "i am",
                "he ' s": "he is",
                "c'mon": "come",
                "lov'd": "love",
                "y'hear": "you hear",
                "'course": "of course",
                "not'a": "not",
                "t'tell": "to tell",
                "m'lord": "my lord"
            }
            sent = contractions.fix(sent)
            sent = sent.lower().strip()
            sent = re.sub(r"('ll)", r" will", sent)
            sent = re.sub(r"('s)", r" \1", sent)
            sent = re.sub(r"[^a-zA-Z0-9',.!?]+", r' ', sent)
            sent = re.sub(r'([,.!?])\1*', r' \1 ', sent)
            sent = re.sub(r'\d+\.?\d*', r' <number> ', sent)
            for k, v in clean_dict.items():
                sent = sent.replace(k, v)
            sent = re.sub(r"(\w)\s*'\s*re", r"\1 are", sent)
            sent = re.sub(r"(\w)\s*'\s*ve", r"\1 have", sent)
            sent = re.sub(r"(\w)\s*'\s*t", r"\1 not", sent)
            sent = re.sub(r"(\w)\s*'\s*d", r"\1 did", sent)
            sent = re.sub(r"(\s+|^)'(\w)", r"\1\2", sent)
            sent = re.sub(r"(\w)'(\s+|$)", r"\1\2", sent)
            sent = re.sub(r" '|' ", r' ', sent)
            sent = re.sub(r'\s+', r' ', sent)
            sent = nltk.word_tokenize(sent)
            return sent

        convs_clean = []
        for conv in tqdm(convs, desc='Clean'):
            convs_clean.append(list(map(_clean, conv)))
        print(f'{len(convs_clean)} conversations cleaned.')
        return convs_clean

    @classmethod
    def make_pairs(cls, convs):
        pairs = []
        for conv in convs:
            for i in range(len(conv) - 1):
                pairs.append((conv[i], conv[i + 1]))
        print(f'{len(pairs)} pairs built.')
        return pairs

    @classmethod
    def filter(cls, pairs, max_len):
        def _filter(sent):
            if len(sent) > max_len:
                return False
            exclude = ',.!?'.split() + ['<number>']
            sent = [x for x in sent if x not in exclude]
            if len(sent) <= 3:
                return False
            return True
        
        pairs_filter = [(x, y) for x, y in pairs if _filter(x) and _filter(y)]
        print('%d pairs retained (%.2f%%).' % \
              (len(pairs_filter), len(pairs_filter) / len(pairs) * 100))
        return pairs_filter

    @classmethod
    def build_vocab(cls, pairs, vocab_size):
        vocab = Vocab()
        for i in range(len(pairs) - 1):
            if pairs[i][1] == pairs[i + 1][0]:
                words = pairs[i][0]
            else:
                words = pairs[i][0] + pairs[i][1]
            for word in words:
                vocab.add_word(word)
        vocab.make_vocab(vocab_size)
        print(f'{len(vocab)} tokens in vocab.')
        return vocab


class Vocab:
    
    def __init__(self):
        self.word_count = {}
        self.index2word = []
        self.word2index = {}
        self.pad_token, self.pad_value = '<pad>', 0
        self.sos_token, self.sos_value = '<sos>', 1
        self.eos_token, self.eos_value = '<eos>', 2
        self.unk_token, self.unk_value = '<unk>', 3
        self.special_tokens = [
            self.pad_token, self.sos_token,
            self.eos_token, self.unk_token,
        ]
        
    def add_word(self, word):
        if word in self.word_count:
            self.word_count[word] += 1
        else:
            self.word_count[word] = 1
            
    def make_vocab(self, size):
        self.word_count = sorted(
            self.word_count, key=self.word_count.get, reverse=True)
        word_size = size - len(self.special_tokens)
        for w in self.special_tokens + self.word_count[:word_size]:
            self.index2word.append(w)
            self.word2index[w] = len(self.index2word) - 1

    def __len__(self):
        return len(self.index2word)
                
    def __getitem__(self, query):
        if isinstance(query, int):
            return self.index2word[query]
        if isinstance(query, str):
            if query in self.word2index:
                return self.word2index[query]
            else:
                return self.unk_value


class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, pairs, vocab):
        self.pairs = pairs
        self.vocab = vocab
    
    def __getitem__(self, index):
        x, y = self.pairs[index]
        x = [self.vocab[w] for w in x]
        y = [self.vocab[w] for w in y] + [self.vocab['<eos>']]
        return x, y
    
    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def collate_fn(batch):
        batch_x, batch_y = [], []
        for x, y in batch:
            batch_x.append(torch.tensor(x))
            batch_y.append(torch.tensor(y))
        batch_x = torch.nn.utils.rnn.pad_sequence(batch_x, padding_value=0)    
        batch_y = torch.nn.utils.rnn.pad_sequence(batch_y, padding_value=0)
        return batch_x, batch_y


def load_embedding(embedding_path, vocab, use_cache=True):
    pickle_path = os.path.join(
        os.path.split(embedding_path)[0], 'embedding.pickle')
    if use_cache and os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            embedding = pickle.load(f)
        print(f'Loaded from {pickle_path}.')
        return embedding

    print('Loading embedding file...')
    df = pd.read_csv(
        embedding_path, sep=' ', index_col=0, header=None, quoting=3)
    print('Load embedding file done.')
    vocab_size = len(vocab)
    embed_size = df.shape[1]
    embedding = np.empty((vocab_size, embed_size), dtype=np.float32)
    
    # Fill specail tokens with random value
    n_special = len(vocab.special_tokens)
    for i in range(n_special):
        embedding[i] = np.random.randn(embed_size)
    # Fill UNK with mean value
    embedding[vocab.unk_value] = df.mean()
    # Fill words
    n_hit = 0
    for i in range(n_special, vocab_size):
        if vocab[i] in df.index:
            embedding[i] = df.iloc[i]
            n_hit += 1
        else:
            embedding[i] = np.random.randn(embed_size)
    hit_ratio = n_hit / vocab_size
    print(f'{n_hit} hits of {vocab_size} words ({hit_ratio * 100:.2}%)')
    
    with open(pickle_path, 'wb') as f:
        pickle.dump(embedding, f)
        print(f'Save data to {pickle_path}.')
    return embedding
