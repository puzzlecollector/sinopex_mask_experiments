import argparse
import logging
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    AutoModel, 
    AdamW, 
    AutoModelForSequenceClassification, 
    get_linear_schedule_with_warmup, 
    get_cosine_schedule_with_warmup
)
from tqdm.auto import tqdm 
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler 
import torch.nn as nn 
import os 
import math 
import time 
import datetime
import string 
import re 
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import f1_score 
from sklearn.utils.class_weight import compute_class_weight


''' intent classifier definition '''
class MeanPooling(nn.Module): 
    def __init__(self):
        super(MeanPooling, self).__init__() 
    def forward(self, last_hidden_state, attention_mask): 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float() 
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1) 
        sum_mask = input_mask_expanded.sum(1) 
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask 
        return mean_embeddings
    
class MultiSampleDropout(nn.Module): 
    def __init__(self, max_dropout_rate, num_samples, classifier): 
        super(MultiSampleDropout, self).__init__() 
        self.dropout = nn.Dropout 
        self.classifier = classifier 
        self.max_dropout_rate = max_dropout_rate 
        self.num_samples = num_samples 
    def forward(self, out): 
        return torch.mean(torch.stack([self.classifier(self.dropout(p=self.max_dropout_rate)(out)) for _, rate in enumerate(np.linspace(0,self.max_dropout_rate, self.num_samples))], dim=0), dim=0)
    
class Classifier(nn.Module): 
    def __init__(self, plm="snunlp/KR-SBERT-V40K-klueNLI-augSTS", num_classes=2): 
        super(Classifier, self).__init__() 
        self.num_classes = num_classes 
        self.config = AutoConfig.from_pretrained(plm) 
        self.model = AutoModel.from_pretrained(plm, config=self.config)
        self.mean_pooler = MeanPooling() 
        self.fc = nn.Linear(self.config.hidden_size, self.num_classes) 
        self._init_weights(self.fc) 
        self.multi_dropout = MultiSampleDropout(0.2, 8, self.fc) 
    def _init_weights(self, module): 
        if isinstance(module, nn.Linear): 
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range) 
            if module.bias is not None: 
                module.bias.data.zero_() 
    def forward(self, input_ids, attention_masks): 
        x = self.model(input_ids, attention_masks)[0] 
        x = self.mean_pooler(x, attention_masks) 
        x = self.multi_dropout(x) 
        return x 

''' kogpt2 chatbot definition ''' 
parser = argparse.ArgumentParser(description='시노봇 based on KoGPT-2')

parser.add_argument('--chat',
                    action='store_true',
                    default=False,
                    help='response generation on given user input')

parser.add_argument('--sentiment',
                    type=str,
                    default='0',
                    help='sentiment for system. 0 is neutral, 1 is negative, 2 is positive.')

parser.add_argument('--model_params',
                    type=str,
                    default='model_chp/model_-last.ckpt',
                    help='model binary for starting chat')

parser.add_argument('--train',
                    action='store_true',
                    default=False,
                    help='for training')

logger = logging.getLogger()
logger.setLevel(logging.INFO)


U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 


class CharDataset(Dataset):
    def __init__(self, chats, max_len=32):
        self._data = chats
        self.first = True
        self.q_token = U_TKN
        self.a_token = S_TKN
        self.sent_token = SENT
        self.bos = BOS
        self.eos = EOS
        self.mask = MASK
        self.pad = PAD
        self.max_len = max_len
        self.tokenizer = TOKENIZER 

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        turn = self._data.iloc[idx]
        q = turn['Q']
        a = turn['A']
        sentiment = str(turn['label'])
        q_toked = self.tokenizer.tokenize(self.q_token + q + \
                                          self.sent_token + sentiment)   
        q_len = len(q_toked)
        a_toked = self.tokenizer.tokenize(self.a_token + a + self.eos)
        a_len = len(a_toked)
        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len
            if a_len <= 0:
                q_toked = q_toked[-(int(self.max_len/2)):]
                q_len = len(q_toked)
                a_len = self.max_len - q_len
                assert a_len > 0
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)
            assert a_len == len(a_toked), f'{a_len} ==? {len(a_toked)}'
        # [mask, mask, ...., mask, ..., <bos>,..A.. <eos>, <pad>....]
        labels = [
            self.mask,
        ] * q_len + a_toked[1:]
        if self.first:
            logging.info("contexts : {}".format(q))
            logging.info("toked ctx: {}".format(q_toked))
            logging.info("response : {}".format(a))
            logging.info("toked response : {}".format(a_toked))
            logging.info('labels {}'.format(labels))
            self.first = False
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        self.max_len
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]
        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]
        return(token_ids, np.array(mask),
               labels_ids)


class KoGPT2Chat(LightningModule):
    def __init__(self, hparams=dict(), **kwargs):
        super(KoGPT2Chat, self).__init__()
        # self.hparams = hparams
        self.hparams.update(hparams) 
        self.neg = -1e18
        self.kogpt2 = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max-len',
                            type=int,
                            default=32,
                            help='max sentence length on input (default: 32)')

        parser.add_argument('--batch-size',
                            type=int,
                            default=96,
                            help='batch size for training (default: 96)')
        parser.add_argument('--lr',
                            type=float,
                            default=5e-5,
                            help='The initial learning rate')
        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')
        return parser

    def forward(self, inputs):
        # (batch, seq_len, hiddens)
        output = self.kogpt2(inputs, return_dict=True)
        return output.logits

    def training_step(self, batch, batch_idx):
        token_ids, mask, label = batch
        out = self(token_ids)
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
        loss = self.loss_function(mask_out.transpose(2, 1), label)
        loss_avg = loss.sum() / mask.sum()
        self.log('train_loss', loss_avg)
        return loss_avg

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        num_train_steps = len(self.train_dataloader()) * self.hparams.max_epochs
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def _collate_fn(self, batch):
        data = [item[0] for item in batch]
        mask = [item[1] for item in batch]
        label = [item[2] for item in batch]
        return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

    def train_dataloader(self):
        data = pd.read_csv('Chatbot_data/ChatbotData.csv')
        self.train_set = CharDataset(data, max_len=self.hparams.max_len)
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, num_workers=2,
            shuffle=True, collate_fn=self._collate_fn)
        return train_dataloader

    def chat(self, sent='0'):
        tok = TOKENIZER
        sent_tokens = tok.tokenize(sent)
        with torch.no_grad():
            while 1:
                q = input('user > ').strip()
                if q == 'quit':
                    break
                a = ''
                while 1:
                    input_ids = torch.LongTensor(tok.encode(U_TKN + q + SENT + sent + S_TKN + a)).unsqueeze(dim=0)
                    pred = self(input_ids)
                    gen = tok.convert_ids_to_tokens(
                        torch.argmax(
                            pred,
                            dim=-1).squeeze().numpy().tolist())[-1]
                    if gen == EOS:
                        break
                    a += gen.replace('▁', ' ')
                print("시노봇 > {}".format(a.strip()))
    
    def generate_point_response(self, q, sent='0'): 
        tok = TOKENIZER
        sent_tokens = tok.tokenize(sent)
        with torch.no_grad():
            a = ''
            while 1:
                input_ids = torch.LongTensor(tok.encode(U_TKN + q + SENT + sent + S_TKN + a)).unsqueeze(dim=0)
                pred = self(input_ids)
                gen = tok.convert_ids_to_tokens(
                    torch.argmax(
                        pred,
                        dim=-1).squeeze().numpy().tolist())[-1]
                if gen == EOS:
                    break
                a += gen.replace('▁', ' ') 
            print("시노봇 > {}".format(a.strip()))

if __name__ == '__main__':  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    # load intent classifier   
    intent_clf = Classifier() 
    checkpoint = torch.load("intent_classifier.pt") 
    intent_clf.load_state_dict(checkpoint) 
    intent_clf.to(device) 
    intent_clf.eval() 
    intent_tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-SBERT-V40K-klueNLI-augSTS") 

    # load trained general chit-chat bot 
    parser = KoGPT2Chat.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    logging.info(args)
    chatbot = KoGPT2Chat.load_from_checkpoint(args.model_params)
    try: 
        while True:
            try: 
                query = input("유저 > ").strip() 
                encoded_input = intent_tokenizer(query, max_length=512, truncation=True, padding="max_length", return_tensors="pt").to(device)
                input_ids = encoded_input["input_ids"] 
                attention_mask = encoded_input["attention_mask"] 
                with torch.no_grad(): 
                    logits = intent_clf(input_ids, attention_mask)  

                logits = nn.Softmax(dim=1)(logits) 
                predicted_class = torch.argmax(logits) 

                if predicted_class == 0: # general 
                    response = chatbot.generate_point_response(query)  
                else: # domain
                    '''
                    response = 병철님_도메인_모델(query)
                    ''' 
                    pass 
                print(response) 
            except Exception as e:
                print(e) 
                print("시노봇 처리중 에러 발생!") 
    except KeyboardInterrupt: 
        print("시노봇 종료하는중...") 
