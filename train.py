import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from dataset import DataBuilder, Dataset, Vocab, load_embedding
from dataset_daily import DataBuilder, Dataset, Vocab, load_embedding
from model import Seq2Seq


class Trainer:
    
    def __init__(self, model, device, train_dataloader, n_epoch, optim, tb_dir, 
                 case_interval, vocab,
                 valid_dataloader=None, test_dataloader=None):
        self.device = device
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.optim = optim
        self.case_interval = case_interval
        self.n_epoch = n_epoch
        self.global_t = None
        self.vocab = vocab
        self.writer = SummaryWriter(tb_dir, flush_secs=1)
        
    def loss_fn(self, input, target):
        input = input.reshape(-1, input.size(-1))
        target = target.reshape(-1)
        loss = torch.nn.functional.cross_entropy(
            input=input, target=target,
            ignore_index=0, reduction='mean')
        return loss
    
    def batch2sents(self, batch):
        sents = []
        for data in batch.tolist():
            for _ in range(data.count(self.vocab.pad_value)):
                data.remove(self.vocab.pad_value)
            if self.vocab.eos_value in data:
                tail = len(data) - data[::-1].index(self.vocab.eos_value)
                data = data[:tail]
            sent = [self.vocab[x] for x in data]
            sents.append(' '.join(sent))
        return sents
    
    def show_case(self, x, y, y_preds):
        post = self.batch2sents(x.t())[1]
        targ = self.batch2sents(y.t())[1]
        pred = y_preds.argmax(dim=2)
        pred = self.batch2sents(pred.t())[1]
        texts = [
            f'[Post] {post}',
            f'[Targ] {targ}',
            f'[Pred] {pred}'
        ]
        texts = '\n\n'.join(texts)
        self.writer.add_text('case', texts, self.global_t)
        
    def train_batch(self, batch):
        x, y = batch[0].to(self.device), batch[1].to(self.device)
        y_preds = self.model(x, y)
        loss = self.loss_fn(input=y_preds, target=y)
        self.model.zero_grad()
        loss.backward()
        self.optim.step()
        self.global_t += 1
        if self.global_t % self.case_interval == 0:
            self.show_case(x, y, y_preds)
        return {'loss': loss.item()}
        
    def overfit_one_batch(self, n_step):
        self.model.train()
        batch = next(iter(self.train_dataloader))
        pbar = tqdm(range(n_step), desc='Overfit')
        self.global_t = 0
        for i in pbar:
            state = self.train_batch(batch)
            pbar.set_postfix(state)
            self.writer.add_scalars('overfit', state, self.global_t)
            
    def fit(self): 
        self.global_t = 0
        for epoch in tqdm(range(1, self.n_epoch + 1), desc='Total'):
            self.train_epoch(epoch)
            if self.valid_dataloader is not None:
                self.valid_epoch(epoch)
            if self.test_dataloader is not None:
                self.test_epoch(epoch)
                
    def train_epoch(self, epoch):
        self.model.train()
        pbar = tqdm(self.train_dataloader, desc=f'Train Epoch {epoch}')
        for batch in pbar:
            state = self.train_batch(batch)
            pbar.set_postfix(state)
            self.writer.add_scalars('train', state, self.global_t)
            

if __name__ == '__main__':
    # data_dir = 'data'
    data_dir = 'data-daily-train'
    embedding_path = 'embedding/glove.42B.300d.txt'
    tb_dir = 'runs/fair'
    case_interval = 10
    gpu_id = 7

    max_len = 30
    vocab_size = 10000
    n_epoch = 30
    learning_rate = 0.001
    batch_size = 64
    embed_size = 300
    hidden_size = 200


    if not torch.cuda.is_available() or gpu_id == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:' + str(gpu_id))

    pairs, vocab = DataBuilder.build(
        data_dir=data_dir,
        max_len=max_len,
        vocab_size=vocab_size,
    )
    dataset = Dataset(pairs, vocab)    
    embedding = load_embedding(embedding_path, vocab)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, 
        collate_fn=dataset.collate_fn,
    )
    model = Seq2Seq(
        embedding=embedding,
        hidden_size=hidden_size, 
        pad_value=vocab.pad_value,
        sos_value=vocab.sos_value
    )
    adam = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )
    trainer = Trainer(
        model=model,
        device=device,
        train_dataloader=train_dataloader,
        n_epoch=n_epoch,
        optim=adam,
        tb_dir=tb_dir,
        case_interval=case_interval,
        vocab=vocab,
    )

    # trainer.overfit_one_batch(500)
    trainer.fit()