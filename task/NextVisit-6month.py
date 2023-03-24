import sys 

sys.path.append('/home/josaphat/Desktop/research/BEHRT')


from common.common import create_folder,load_obj
from dataLoader.utils import seq_padding,code2index, position_idx, index_seg
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset
import os
import torch
import torch.nn as nn
import pytorch_pretrained_bert as Bert
from model.utils import age_vocab
from model import optimiser
import sklearn.metrics as skm
import math
from torch.utils.data.dataset import Dataset
import random
import numpy as np
import torch
import time

file_config = {
    'vocab':'token2idx-added',  # vocab token2idx idx2token
    'train': './behrt_format_mimic4ed_month_based_train/',
    'test': './behrt_format_mimic4ed_month_based_test/',
}

optim_config = {
    'lr': 3e-5,
    'warmup_proportion': 0.1,
    'weight_decay': 0.01
}

global_params = {
    'batch_size': 64, #64
    'gradient_accumulation_steps': 1,
    'device': 'cuda:0', # originally is cuda:0
    'output_dir': 'exp-nextvisit',  # output dir
    'best_name': 'minvisit3-monthbased-best', # output model name
    'save_model': True,
    'max_len_seq': 64, # originally is 100, is it okay to change this to match max_seq_len of MLM.py model?
    'max_age': 110,
    'month': 1,
    'age_symbol': None,
    'min_visit': 3 # originally is 5, should change this to 3?
}

pretrainModel = 'exp-model/minvisit3-monthbased-model' 

create_folder(global_params['output_dir'])



BertVocab = load_obj(file_config['vocab'])
ageVocab, _ = age_vocab(max_age=global_params['max_age'], mon=global_params['month'], symbol=global_params['age_symbol'])



def format_label_vocab(token2idx):
    token2idx = token2idx.copy()
    del token2idx['PAD']
    del token2idx['SEP']
    del token2idx['CLS']
    del token2idx['MASK']
    token = list(token2idx.keys())
    labelVocab = {}
    for i,x in enumerate(token):
        labelVocab[x] = i
    return labelVocab

Vocab_diag = format_label_vocab(BertVocab['token2idx'])



model_config = {
    'vocab_size': len(BertVocab['token2idx'].keys()), # number of disease + symbols for word embedding
    'hidden_size': 288, # word embedding and seg embedding hidden size
    'seg_vocab_size': 2, # number of vocab for seg embedding
    'age_vocab_size': len(ageVocab.keys()), # number of vocab for age embedding
    'max_position_embedding': global_params['max_len_seq'], # maximum number of tokens
    'hidden_dropout_prob': 0.2, # dropout rate
    'num_hidden_layers': 6, # number of multi-head attention layers required
    'num_attention_heads': 12, # number of attention heads
    'attention_probs_dropout_prob': 0.22, # multi-head attention dropout rate
    'intermediate_size': 512, # the size of the "intermediate" layer in the transformer encoder
    'hidden_act': 'gelu', # The non-linear activation function in the encoder and the pooler "gelu", 'relu', 'swish' are supported
    'initializer_range': 0.02, # parameter weight initializer range
}

feature_dict = {
    'age': True,
    'seg': True,
    'posi': True
}




class NextVisit(Dataset):
    def __init__(self, token2idx, diag2idx, age2idx,dataframe, max_len, max_age=110, min_visit=5):
        # dataframe preproecssing
        # filter out the patient with number of visits less than min_visit
        self.vocab = token2idx
        self.label_vocab = diag2idx
        self.max_len = max_len
        self.code = dataframe.code
        self.age = dataframe.age
        self.label = dataframe.label
        self.patid = dataframe.patid

        self.age2idx = age2idx

    def __getitem__(self, index):
        """
        return: age, code, position, segmentation, mask, label
        """
        # cut data
        age = self.age[index]
        code = self.code[index]
        label = self.label[index]
        patid = self.patid[index]

        # extract data
        age = age[(-self.max_len+1):]
        code = code[(-self.max_len+1):]

        # avoid data cut with first element to be 'SEP'
        if code[0] != 'SEP':
            code = np.append(np.array(['CLS']), code)
            age = np.append(np.array(age[0]), age)
        else:
            code[0] = 'CLS'

        # mask 0:len(code) to 1, padding to be 0
        mask = np.ones(self.max_len)
        mask[len(code):] = 0

        # pad age sequence and code sequence
        age = seq_padding(age, self.max_len, token2idx=self.age2idx)

        tokens, code = code2index(code, self.vocab)
        _, label = code2index(label, self.label_vocab)

        # get position code and segment code
        tokens = seq_padding(tokens, self.max_len)
        position = position_idx(tokens)
        segment = index_seg(tokens)

        # pad code and label
        code = seq_padding(code, self.max_len, symbol=self.vocab['PAD'])
        label = seq_padding(label, self.max_len, symbol=-1)

        return torch.LongTensor(age), torch.LongTensor(code), torch.LongTensor(position), torch.LongTensor(segment), \
               torch.LongTensor(mask), torch.LongTensor(label), torch.LongTensor([int(patid)])

    def __len__(self):
        return len(self.code)
    



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

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, segment, age
    """

    def __init__(self, config, feature_dict):
        super(BertEmbeddings, self).__init__()
        self.feature_dict = feature_dict
        
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.segment_embeddings = nn.Embedding(config.seg_vocab_size, config.hidden_size)
        self.age_embeddings = nn.Embedding(config.age_vocab_size, config.hidden_size)
        self.posi_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size).\
            from_pretrained(embeddings=self._init_posi_embedding(config.max_position_embeddings, config.hidden_size))

        self.LayerNorm = Bert.modeling.BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, word_ids, age_ids=None, seg_ids=None, posi_ids=None, age=True):
        if seg_ids is None:
            seg_ids = torch.zeros_like(word_ids)
        if age_ids is None:
            age_ids = torch.zeros_like(word_ids)
        if posi_ids is None:
            posi_ids = torch.zeros_like(word_ids)

        word_embed = self.word_embeddings(word_ids)
        segment_embed = self.segment_embeddings(seg_ids)
        age_embed = self.age_embeddings(age_ids)
        posi_embeddings = self.posi_embeddings(posi_ids)
        
        embeddings = word_embed
        
        if self.feature_dict['age']:
            embeddings = embeddings + age_embed
        if self.feature_dict['seg']:
            embeddings = embeddings + segment_embed
        if self.feature_dict['posi']:
            embeddings = embeddings + posi_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def _init_posi_embedding(self, max_position_embedding, hidden_size):
        def even_code(pos, idx):
            return np.sin(pos/(10000**(2*idx/hidden_size)))

        def odd_code(pos, idx):
            return np.cos(pos/(10000**(2*idx/hidden_size)))

        # initialize position embedding table
        lookup_table = np.zeros((max_position_embedding, hidden_size), dtype=np.float32)

        # reset table parameters with hard encoding
        # set even dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(0, hidden_size, step=2):
                lookup_table[pos, idx] = even_code(pos, idx)
        # set odd dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(1, hidden_size, step=2):
                lookup_table[pos, idx] = odd_code(pos, idx)

        return torch.tensor(lookup_table)

class BertModel(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config, feature_dict):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config, feature_dict)
        self.encoder = Bert.modeling.BertEncoder(config=config)
        self.pooler = Bert.modeling.BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, age_ids=None, seg_ids=None, posi_ids=None, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if age_ids is None:
            age_ids = torch.zeros_like(input_ids)
        if seg_ids is None:
            seg_ids = torch.zeros_like(input_ids)
        if posi_ids is None:
            posi_ids = torch.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, age_ids, seg_ids, posi_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output

class BertForMultiLabelPrediction(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config, num_labels, feature_dict):
        super(BertForMultiLabelPrediction, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config, feature_dict)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, age_ids=None, seg_ids=None, posi_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, age_ids ,seg_ids, posi_ids, attention_mask,
                                     output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.MultiLabelSoftMarginLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss, logits
        else:
            return logits
        


data = pd.read_parquet(file_config['train']).reset_index(drop=True)
# how do i set the label? currently i set it to an array of size 1, containing 1
data['label'] = data.label.apply(lambda x: list(set(x)))
Dset = NextVisit(token2idx=BertVocab['token2idx'], diag2idx=Vocab_diag, age2idx=ageVocab,dataframe=data, max_len=global_params['max_len_seq'])
trainload = DataLoader(dataset=Dset, batch_size=global_params['batch_size'], shuffle=True, num_workers=3)



data = pd.read_parquet(file_config['test']).reset_index(drop=True)
data['label'] = data.label.apply(lambda x: list(set(x)))
Dset = NextVisit(token2idx=BertVocab['token2idx'], diag2idx=Vocab_diag, age2idx=ageVocab,dataframe=data, max_len=global_params['max_len_seq'])
testload = DataLoader(dataset=Dset, batch_size=global_params['batch_size'], shuffle=False, num_workers=3)



conf = BertConfig(model_config)
model = BertForMultiLabelPrediction(conf, len(Vocab_diag.keys()), feature_dict)



# load pretrained model and update weights
pretrained_dict = torch.load(pretrainModel)
model_dict = model.state_dict()
# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict) 
# 3. load the new state dict
model.load_state_dict(model_dict)



model = model.to(global_params['device'])
optim = optimiser.adam(params=list(model.named_parameters()), config=optim_config)



import sklearn
def precision(logits, label):
    sig = nn.Sigmoid()
    output=sig(logits)
    label, output=label.cpu(), output.detach().cpu()
    tempprc= sklearn.metrics.average_precision_score(label.numpy(),output.numpy(), average='samples')
    return tempprc, output, label

def precision_test(logits, label):
    sig = nn.Sigmoid()
    output=sig(logits)
    tempprc= sklearn.metrics.average_precision_score(label.numpy(),output.numpy(), average='samples')
#     roc = sklearn.metrics.roc_auc_score()
    return tempprc, output, label

def auroc_test(logits, label):
    sig = nn.Sigmoid()
    output=sig(logits)
    tempprc= sklearn.metrics.roc_auc_score(label.numpy(),output.numpy(), average='samples')
#     roc = sklearn.metrics.roc_auc_score()
    return tempprc



from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer(classes=list(Vocab_diag.values()))
mlb.fit([[each] for each in list(Vocab_diag.values())])



def train(e):
    model.train()
    tr_loss = 0
    temp_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    cnt = 0
    for step, batch in enumerate(trainload):
        cnt +=1
        age_ids, input_ids, posi_ids, segment_ids, attMask, targets, _ = batch
        targets = torch.tensor(mlb.transform(targets.numpy()), dtype=torch.float32)
        
        age_ids = age_ids.to(global_params['device'])
        input_ids = input_ids.to(global_params['device'])
        posi_ids = posi_ids.to(global_params['device'])
        segment_ids = segment_ids.to(global_params['device'])
        attMask = attMask.to(global_params['device'])
        targets = targets.to(global_params['device'])
        
        loss, logits = model(input_ids, age_ids, segment_ids, posi_ids,attention_mask=attMask, labels=targets)
        
        if global_params['gradient_accumulation_steps'] >1:
            loss = loss/global_params['gradient_accumulation_steps']
        loss.backward()
        
        temp_loss += loss.item()
        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        
        if step % 200==0:
            prec, a, b = precision(logits, targets)
            print("epoch: {}\t| Cnt: {}\t| Loss: {}\t| precision: {}".format(e, cnt,temp_loss/2000, prec))
            temp_loss = 0
            
        
        if (step + 1) % global_params['gradient_accumulation_steps'] == 0:
            optim.step()
            optim.zero_grad()

def evaluation():
    model.eval()
    y = []
    y_label = []
    for step, batch in enumerate(testload):
        model.eval()
        age_ids, input_ids, posi_ids, segment_ids, attMask, targets, _ = batch
        targets = torch.tensor(mlb.transform(targets.numpy()), dtype=torch.float32)
        
        age_ids = age_ids.to(global_params['device'])
        input_ids = input_ids.to(global_params['device'])
        posi_ids = posi_ids.to(global_params['device'])
        segment_ids = segment_ids.to(global_params['device'])
        attMask = attMask.to(global_params['device'])
        targets = targets.to(global_params['device'])
        
        with torch.no_grad():
            loss, logits = model(input_ids, age_ids, segment_ids, posi_ids,attention_mask=attMask, labels=targets)
        logits = logits.cpu()
        targets = targets.cpu()

        y_label.append(targets)
        y.append(logits)

    y_label = torch.cat(y_label, dim=0)
    y = torch.cat(y, dim=0)

    tempprc, output, label = precision_test(y, y_label)
    auroc = auroc_test(y, y_label)
    return tempprc, auroc

import warnings
warnings.filterwarnings(action='ignore')
optim_config = {
    'lr': 3e-6,
    'warmup_proportion': 0.1
}
optim = optimiser.adam(params=list(model.named_parameters()), config=optim_config)

best_pre = 0.512
for e in range(50):
    train(e)
    auc, roc= evaluation()
    if auc >best_pre:
        # Save a trained model
        print("** ** * Saving fine - tuned model ** ** * ")
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(global_params['output_dir'],global_params['best_name'])
        create_folder(global_params['output_dir'])
        if global_params['save_model']:
            torch.save(model_to_save.state_dict(), output_model_file)
        best_pre = auc
    print('precision : {}, auroc: {},'.format(auc, roc))