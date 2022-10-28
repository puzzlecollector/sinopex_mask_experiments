import numpy as np 
import pandas as pd 
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
import torch 
import os 
import math 
import time 
import datetime
import string 
import re 
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import f1_score 
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

df = pd.read_csv("시노봇학습데이터.csv") 

questions = df["Questions"].values 
labels = df["Labels"].values 

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
   
class WeightedFocalLoss(nn.Module): 
    def __init__(self, alpha, gamma=2): 
        super(WeightedFocalLoss, self).__init__() 
        self.alpha = alpha 
        self.device = torch.device("cuda") 
        self.alpha = self.alpha.to(self.device) 
        self.gamma = gamma 
    def forward(self, inputs, targets): 
        CE_loss = nn.CrossEntropyLoss()(inputs, targets) 
        targets = targets.type(torch.long) 
        at = self.alpha.gather(0, targets.data.view(-1)) 
        pt = torch.exp(-CE_loss) 
        F_loss = at * (1-pt)**self.gamma * CE_loss
        return F_loss.mean() 
      
tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-SBERT-V40K-klueNLI-augSTS") 
input_ids, attention_masks = [], [] 
for i in tqdm(range(len(questions))): 
    encoded_input = tokenizer(questions[i], max_length=512, truncation=True, padding="max_length") 
    input_ids.append(encoded_input["input_ids"]) 
    attention_masks.append(encoded_input["attention_mask"]) 

input_ids = torch.tensor(input_ids, dtype=int) 
attention_masks = torch.tensor(attention_masks, dtype=int) 
labels = torch.tensor(labels, dtype=int)

print(input_ids.shape, attention_masks.shape, labels.shape) 

def flat_accuracy(preds, labels): 
    pred_flat = np.argmax(preds, axis=1).flatten() 
    labels_flat = labels.flatten() 
    return np.sum(pred_flat==labels_flat) / len(labels_flat) 

all_train_losses, all_val_losses = [], [] 
all_train_accuracies, all_val_accuracies = [], [] 

skf = StratifiedKFold(n_splits=4, random_state=42, shuffle=True) 
for idx, (train_idx, valid_idx) in enumerate(skf.split(input_ids,labels)): 
    if idx > 0: 
        break 
    print("="*20 + f" KFOLD {idx+1} " + "="*20)
    train_input_ids, valid_input_ids = input_ids[train_idx], input_ids[valid_idx] 
    train_attn_masks, valid_attn_masks = attention_masks[train_idx], attention_masks[valid_idx] 
    train_labels, valid_labels = labels[train_idx], labels[valid_idx] 
    
    print(train_labels.shape, valid_labels.shape) 
    
    
    train_data = TensorDataset(train_input_ids, train_attn_masks, train_labels) 
    train_sampler = RandomSampler(train_data) 
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=16) 
    
    val_data = TensorDataset(valid_input_ids, valid_attn_masks, valid_labels) 
    val_sampler = SequentialSampler(val_data) 
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=16) 
    
    train_losses, val_losses = [], [] 
    train_accuracies, val_accuracies = [], [] 
    
    model = Classifier() 
    model.cuda() 
    optimizer = AdamW(model.parameters(),lr=2e-5,eps=1e-8) 
    epochs = 5
    total_steps = len(train_dataloader) * epochs 
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps = total_steps) 
    device = torch.device("cuda") 
    loss_func = nn.CrossEntropyLoss() 
    '''
    # training with weighted focal loss 
    class_weights = compute_class_weight(class_weight = "balanced", classes = np.unique(train_labels), y=np.array(train_labels)) 
    class_weights = torch.tensor(class_weights, dtype=torch.float) 
    loss_func = WeightedFocalLoss(alpha=class_weights) 
    '''
    model.zero_grad()
    for epoch_i in tqdm(range(epochs), desc="Epochs", position=0, leave=True, total=epochs):
        train_loss, train_accuracy = 0, 0 
        model.train() 
        with tqdm(train_dataloader, unit="batch") as tepoch: 
            for step, batch in enumerate(tepoch): 
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch 
                outputs = model(b_input_ids, b_input_mask) 
                logits = outputs 
                loss = loss_func(logits, b_labels) 
                train_loss += loss.item() 
                logits_cpu = logits.detach().cpu().numpy() 
                label_ids = b_labels.detach().cpu().numpy() 
                cur_accuracy = flat_accuracy(logits_cpu, label_ids) 
                train_accuracy += cur_accuracy  
                
                loss.backward() 
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
                optimizer.step() 
                scheduler.step() 
                model.zero_grad() 
                tepoch.set_postfix(loss=train_loss/(step+1), accuracy=100.0 * train_accuracy / (step+1)) 
                time.sleep(0.1) 
            avg_train_loss = train_loss / len(train_dataloader) 
            avg_train_accuracy = train_accuracy / len(train_dataloader) 
            print("average train loss : {}".format(avg_train_loss)) 
            print("average train accuracy : {}".format(avg_train_accuracy)) 
            train_losses.append(avg_train_loss) 
            train_accuracies.append(avg_train_accuracy) 
        ## validate ## 
        val_loss, val_accuracy = 0, 0 
        val_f1 = 0 
        model.eval() 
        for step, batch in tqdm(enumerate(val_dataloader), desc="validating", position=0, leave=True, total=len(val_dataloader)): 
            batch = tuple(t.to(device) for t in batch) 
            b_input_ids, b_input_masks, b_labels = batch 
            with torch.no_grad(): 
                outputs = model(b_input_ids, b_input_masks) 
            logits = outputs 
            loss = loss_func(logits, b_labels) 
            val_loss += loss.item() 
            logits_cpu = logits.detach().cpu().numpy()
            label_ids = b_labels.detach().cpu().numpy() 
            cur_accuracy = flat_accuracy(logits_cpu, label_ids) 
            val_accuracy += cur_accuracy              
            pred_labels = np.argmax(logits_cpu, axis=1) 
            true_labels = label_ids 
            
            
        avg_val_loss = val_loss / len(val_dataloader) 
        avg_val_accuracy = val_accuracy / len(val_dataloader)  
        print("average val loss : {}".format(avg_val_loss)) 
        print("average val accuracy: {}".format(avg_val_accuracy))  
        val_losses.append(avg_val_loss) 
        val_accuracies.append(avg_val_accuracy) 
        if np.min(val_losses) == val_losses[-1]: 
            torch.save(model.state_dict(), f"test_model_KFOLD{idx+1}_val_loss:{avg_val_loss}_val_accuracy:{avg_val_accuracy}.pt") 
    
    all_train_losses.append(train_losses) 
    all_val_losses.append(val_losses) 
    
    all_train_accuracies.append(train_accuracies) 
    all_val_accuracies.append(val_accuracies)      
  

