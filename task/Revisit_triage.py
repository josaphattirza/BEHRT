import sys
from matplotlib import pyplot as plt 

sys.path.append('/home/josaphat/Desktop/research/BEHRT')

from torch.utils.data import DataLoader
import pandas as pd
from common.common import create_folder,H5Recorder
import numpy as np
from torch.utils.data.dataset import Dataset
import os
import torch
import torch.nn as nn
import pytorch_pretrained_bert as Bert

from model import optimiser
import sklearn.metrics as skm
import math
from torch.utils.data.dataset import Dataset
import random
import numpy as np
import torch
import time
from sklearn.metrics import roc_auc_score
from common.common import load_obj
from model.utils import age_vocab
from dataLoader.Disposition_triage_med import NextVisit
from model.Disposition_triage_med import BertForMultiLabelPrediction
import warnings
warnings.filterwarnings(action='ignore')

file_config = {
    'vocab':'token2idx-added',  # vocab token2idx idx2token
    'med_vocab' : 'med2idx', 
    'triage_vocab' : 'triage2idx',
    'train': './behrt_triage_revisit_disposition_med_month_based_train/',
    'test': './behrt_triage_revisit_disposition_med_month_based_test/',
}

optim_config = {
    'lr': 3e-5, # originally 3e-5
    'warmup_proportion': 0.1,
    'weight_decay': 0.01
}

global_params = {
    'batch_size': 64,
    'gradient_accumulation_steps': 1, # originally 1
    'device': 'cuda:0',
    'output_dir': 'finetune-revisit-triage',  # output dir
    'best_name': 'finetune-revisit-triage-monthbased-best', # output model name
    'max_len_seq': 64, # originally is 100, ?
    'max_age': 110,
    'age_year': False,
    'age_symbol': None,
    'min_visit': 3 # originally is 5
}

pretrain_model_path = 'triage-med-MLM/triage-med-MLM-minvisit3-monthbased'  # pretrained MLM path

BertVocab = load_obj(file_config['vocab'])
med_BertVocab = load_obj(file_config['med_vocab'])
triage_BertVocab = load_obj(file_config['triage_vocab'])
ageVocab, _ = age_vocab(max_age=global_params['max_age'], symbol=global_params['age_symbol'])


# # re-format label token
# def format_label_vocab(token2idx):
#     token2idx = token2idx.copy()
#     del token2idx['PAD']
#     del token2idx['SEP']
#     del token2idx['CLS']
#     del token2idx['MASK']
#     token = list(token2idx.keys())
#     labelVocab = {}
#     for i,x in enumerate(token):
#         labelVocab[x] = i
#     return labelVocab


# OWN LABEL VOCAB , since we want to predict disposition
labelKey = ["UNK","Yes","No"]
labelVocab = {}
for i,x in enumerate(labelKey):
    labelVocab[x] = i


model_config = {
    'vocab_size': len(BertVocab['token2idx'].keys()), # number of disease + symbols for word embedding
    'med_vocab_size': len(med_BertVocab['med2idx'].keys()), # OWN EMBEDDINGS
    'triage_vocab_size': len(triage_BertVocab['triage2idx'].keys()), # OWN EMBEDDINGS



    'hidden_size': 288, # word embedding and seg embedding hidden size
    'seg_vocab_size': 2, # number of vocab for seg embedding
    'age_vocab_size': len(ageVocab.keys()), # number of vocab for age embedding
    'max_position_embedding': global_params['max_len_seq'], # maximum number of tokens
    'hidden_dropout_prob': 0.1, # dropout rate
    'num_hidden_layers': 6, # number of multi-head attention layers required
    'num_attention_heads': 12, # number of attention heads
    'attention_probs_dropout_prob': 0.1, # multi-head attention dropout rate
    'intermediate_size': 512, # the size of the "intermediate" layer in the transformer encoder
    'hidden_act': 'gelu', # The non-linear activation function in the encoder and the pooler "gelu", 'relu', 'swish' are supported
    'initializer_range': 0.02, # parameter weight initializer range
}

feature_dict = {
    'word':True,
    'med':True, # OWN EMBEDDINGS
    'triage':True, # OWN EMBEDDINGS
    'seg':True,
    'age':True,
    'position': True
}



class BertConfig(Bert.modeling.BertConfig):
    def __init__(self, config):
        super(BertConfig, self).__init__(
            vocab_size_or_config_json_file=config.get('vocab_size'),
            hidden_size=config['hidden_size'],
            num_hidden_layers=config.get('num_hidden_layers'),
            num_attention_heads=config.get('num_attention_heads'),
            intermediate_size=config.get('intermediate_size'),
            hidden_act=config.get('hidden_act'),
            hidden_dropout_prob=config.get('hidden_dropout_prob'),
            attention_probs_dropout_prob=config.get('attention_probs_dropout_prob'),
            max_position_embeddings = config.get('max_position_embedding'),
            initializer_range=config.get('initializer_range'),
        )
        self.seg_vocab_size = config.get('seg_vocab_size')
        self.age_vocab_size = config.get('age_vocab_size')
        self.med_vocab_size = config.get('med_vocab_size')  # OWN EMBEDDINGS
        self.triage_vocab_size = config.get('triage_vocab_size')  # OWN EMBEDDINGS



train = pd.read_parquet(file_config['train'])
Dset = NextVisit(token2idx=BertVocab['token2idx'], label2idx=labelVocab, age2idx=ageVocab, 
                 med_token2idx=med_BertVocab['med2idx'], triage2idx=triage_BertVocab['triage2idx'],
                 dataframe=train, max_len=global_params['max_len_seq'])
trainload = DataLoader(dataset=Dset, batch_size=global_params['batch_size'], shuffle=True, num_workers=1)



test = pd.read_parquet(file_config['test'])
Dset = NextVisit(token2idx=BertVocab['token2idx'], label2idx=labelVocab, age2idx=ageVocab, 
                 med_token2idx=med_BertVocab['med2idx'], triage2idx=triage_BertVocab['triage2idx'],
                 dataframe=test, max_len=global_params['max_len_seq'])
testload = DataLoader(dataset=Dset, batch_size=global_params['batch_size'], shuffle=False, num_workers=1)



# del model
conf = BertConfig(model_config)
model = BertForMultiLabelPrediction(conf, num_labels=len(labelVocab.keys()), feature_dict=feature_dict)



def load_model(path, model):
    # load pretrained model and update weights
    pretrained_dict = torch.load(path)
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model

mode = load_model(pretrain_model_path, model)


model = model.to(global_params['device'])
optim = optimiser.adam(params=list(model.named_parameters()), config=optim_config)


import sklearn
def precision(logits, label):
    sig = nn.Sigmoid()
    output=sig(logits)
    label, output=label.cpu(), output.detach().cpu()
    tempprc= sklearn.metrics.average_precision_score(label.numpy(),output.numpy(), average='samples')
    print('tempprc', tempprc)
    return tempprc, output, label

def precision_test(logits, label):
    sig = nn.Sigmoid()
    output=sig(logits)
    tempprc= sklearn.metrics.average_precision_score(label.numpy(),output.numpy(), average='samples')
    roc = sklearn.metrics.roc_auc_score(label.numpy(),output.numpy(), average='samples')
    print('auroc', roc)

    sig = nn.Sigmoid()
    output = sig(logits).numpy()

    # fpr, tpr, thresholds = sklearn.metrics.roc_curve(label.numpy().ravel(), output.ravel())
    # plt.plot(fpr, tpr)
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.show()
    
    return tempprc, roc, output, label,


from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer(classes=list(labelVocab.values()))
mlb.fit([[each] for each in list(labelVocab.values())])



def train(e):
    model.train()
    tr_loss = 0
    temp_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    cnt = 0
    for step, batch in enumerate(trainload):
        cnt +=1
        age_ids, input_ids, posi_ids, segment_ids, attMask, targets, _ , med_input_ids, triage_input_ids = batch
        
        # BELOW

        targets = torch.tensor(mlb.transform(targets.numpy()), dtype=torch.float32)


        age_ids = age_ids.to(global_params['device'])
        med_input_ids = med_input_ids.to(global_params['device']) # OWN EMBEDDINGS
        triage_input_ids = triage_input_ids.to(global_params['device']) # OWN EMBEDDINGS

        input_ids = input_ids.to(global_params['device'])
        posi_ids = posi_ids.to(global_params['device'])
        segment_ids = segment_ids.to(global_params['device'])
        attMask = attMask.to(global_params['device'])
        targets = targets.to(global_params['device'])
        
        loss, logits = model(input_ids = input_ids, 
                             age_ids = age_ids, 
                             seg_ids = segment_ids, 
                             posi_ids = posi_ids,
                             attention_mask=attMask, labels=targets,
                             med_input_ids = med_input_ids,
                             triage_input_ids = triage_input_ids)
        
        if global_params['gradient_accumulation_steps'] >1:
            loss = loss/global_params['gradient_accumulation_steps']
        loss.backward()
        
        temp_loss += loss.item()
        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        
        if step % 200==0:
            prec, a, b = precision(logits, targets)
            print("epoch: {}\t| Cnt: {}\t| Loss: {}\t| precision: {}".format(e, cnt,temp_loss/500, prec))
            temp_loss = 0
        
        if (step + 1) % global_params['gradient_accumulation_steps'] == 0:
            optim.step()
            optim.zero_grad()
    
    print('final step amount: ', step)


def evaluation():
    model.eval()
    y = []
    y_label = []
    tr_loss = 0
    for step, batch in enumerate(testload):
        model.eval()
        age_ids, input_ids, posi_ids, segment_ids, attMask, targets, _ , med_input_ids, triage_input_ids = batch
        targets = torch.tensor(mlb.transform(targets.numpy()), dtype=torch.float32)
        
        age_ids = age_ids.to(global_params['device'])
        med_input_ids = med_input_ids.to(global_params['device']) # OWN EMBEDDINGS
        triage_input_ids = triage_input_ids.to(global_params['device']) # OWN EMBEDDINGS

        input_ids = input_ids.to(global_params['device'])
        posi_ids = posi_ids.to(global_params['device'])
        segment_ids = segment_ids.to(global_params['device'])
        attMask = attMask.to(global_params['device'])
        targets = targets.to(global_params['device'])
        
        with torch.no_grad():
            loss, logits = model(input_ids = input_ids, 
                             age_ids = age_ids, 
                             seg_ids = segment_ids, 
                             posi_ids = posi_ids,
                             attention_mask=attMask, labels=targets,
                             med_input_ids = med_input_ids,
                             triage_input_ids = triage_input_ids)
        logits = logits.cpu()
        targets = targets.cpu()
        
        tr_loss += loss.item()

        y_label.append(targets)
        y.append(logits)

    y_label = torch.cat(y_label, dim=0)
    y = torch.cat(y, dim=0)

    aps, roc, output, label = precision_test(y, y_label)
    return aps, roc, tr_loss



best_pre = 0.0

# # originally epoch is 50
# for e in range(50): 
for e in range(20):

    train(e)
    aps, roc, test_loss = evaluation()
    if aps > best_pre:
        # Save a trained model
        print("** ** * Saving fine - tuned model ** ** * ")
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(global_params['output_dir'],global_params['best_name'])
        create_folder(global_params['output_dir'])

        torch.save(model_to_save.state_dict(), output_model_file)
        best_pre = aps
    print('aps : {}'.format(aps))