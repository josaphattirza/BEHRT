import sys 
import os

sys.path.append('/home/josaphat/Desktop/research/BEHRT')

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from common.common import create_folder
from common.pytorch import load_model
import pytorch_pretrained_bert as Bert
from model.utils import age_vocab
from common.common import load_obj
from dataLoader.MLM_med import MLMLoader
from torch.utils.data import DataLoader
import pandas as pd
from model.MLM_med import BertForMaskedLM
from model.optimiser import adam
import sklearn.metrics as skm
import numpy as np
import torch
import time
import torch.nn as nn
import gc

gc.collect()
torch.cuda.empty_cache()
gc.collect()


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
        
class TrainConfig(object):
    def __init__(self, config):
        self.batch_size = config.get('batch_size')
        self.use_cuda = config.get('use_cuda')
        self.max_len_seq = config.get('max_len_seq')
        self.train_loader_workers = config.get('train_loader_workers')
        self.test_loader_workers = config.get('test_loader_workers')
        self.device = config.get('device')
        self.output_dir = config.get('output_dir')
        self.output_name = config.get('output_name')
        self.best_name = config.get('best_name')

file_config = {
    'vocab':'token2idx-added',  # vocabulary idx2token, token2idx
    'med_vocab' : 'token2idx-added-med', 
    'data': './behrt_format_mimic4ed/',  # formated data 
    'model_path': 'josa_model', # where to save model
    'model_name': 'test_model', # model name
    'file_name': 'test_model',  # log path
}
create_folder(file_config['model_path'])

global_params = {
    'max_seq_len': 64,
    'max_age': 110,
    'month': 1,
    'age_symbol': None,
    'min_visit': 3, # original is 5
    'gradient_accumulation_steps': 1
}

optim_param = {
    'lr': 3e-5,
    'warmup_proportion': 0.1,
    'weight_decay': 0.01
}

train_params = {
    'batch_size': 32, # original is 256
    'use_cuda': False,
    'max_len_seq': global_params['max_seq_len'],
    'device': 'cpu'
}

BertVocab = load_obj(file_config['vocab'])
med_BertVocab = load_obj(file_config['med_vocab'])
ageVocab, _ = age_vocab(max_age=global_params['max_age'], mon=global_params['month'], symbol=global_params['age_symbol'])
# print(ageVocab)

data = pd.read_parquet(file_config['data'])
# remove patients with visits less than min visit
# print(data)
data['length'] = data['icd_code'].apply(lambda x: len([i for i in range(len(x)) if x[i] == 'SEP']))
data = data[data['length'] >= global_params['min_visit']]
data = data.reset_index(drop=True)

# print(data)
Dset = MLMLoader(dataframe = data, token2idx = BertVocab['token2idx'], age2idx = ageVocab, med_token2idx=med_BertVocab['token2idx'], max_len=train_params['max_len_seq'], code='icd_code', age='age_on_admittance', med='icd_code')

trainload = DataLoader(dataset=Dset, batch_size=train_params['batch_size'], shuffle=True, num_workers=3)

model_config = {
    'vocab_size': len(BertVocab['token2idx'].keys()), # number of disease + symbols for word embedding
    'med_vocab_size': len(med_BertVocab['token2idx'].keys()), # OWN EMBEDDINGS

    'hidden_size': 288, # word embedding and seg embedding hidden size
    'seg_vocab_size': 2, # number of vocab for seg embedding
    'age_vocab_size': len(ageVocab.keys()), # number of vocab for age embedding
    'max_position_embedding': train_params['max_len_seq'], # maximum number of tokens
    'hidden_dropout_prob': 0.1, # dropout rate
    'num_hidden_layers': 6, # number of multi-head attention layers required
    'num_attention_heads': 12, # number of attention heads
    'attention_probs_dropout_prob': 0.1, # multi-head attention dropout rate
    'intermediate_size': 512, # the size of the "intermediate" layer in the transformer encoder
    'hidden_act': 'gelu', # The non-linear activation function in the encoder and the pooler "gelu", 'relu', 'swish' are supported
    'initializer_range': 0.02, # parameter weight initializer range
}

print(model_config['vocab_size'])
print(model_config['max_position_embedding'])

conf = BertConfig(model_config)
model = BertForMaskedLM(conf)

model = model.to(train_params['device'])
optim = adam(params=list(model.named_parameters()), config=optim_param)

def cal_acc(label, pred):
    logs = nn.LogSoftmax()
    label=label.cpu().numpy()
    ind = np.where(label!=-1)[0]
    truepred = pred.detach().cpu().numpy()
    truepred = truepred[ind]
    truelabel = label[ind]
    truepred = logs(torch.tensor(truepred))
    outs = [np.argmax(pred_x) for pred_x in truepred.numpy()]
    precision = skm.precision_score(truelabel, outs, average='micro')
    return precision

def train(e, loader):
    tr_loss = 0
    temp_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    cnt= 0
    start = time.time()

    for step, batch in enumerate(loader):
        cnt +=1
        batch = tuple(t.to(train_params['device']) for t in batch)

        # age_ids, input_ids, posi_ids, segment_ids, attMask, masked_label = batch
        age_ids, input_ids, posi_ids, segment_ids, attMask, masked_label, \
            med_input_ids, med_posi_ids, med_segments_ids, med_attMask, med_masked_label = batch
        
        # loss, pred, label = model(input_ids, age_ids, segment_ids, posi_ids,attention_mask=attMask, masked_lm_labels=masked_label)
        loss, pred, label, = model(input_ids=input_ids, med_input_ids = med_input_ids, age_ids=age_ids, seg_ids=segment_ids, posi_ids=posi_ids,attention_mask=attMask, masked_lm_labels=masked_label)

        # print('loss', loss)
        # print('pred', pred)
        # print('label', label)

        if global_params['gradient_accumulation_steps'] >1:
            loss = loss/global_params['gradient_accumulation_steps']
        loss.backward()
        
        temp_loss += loss.item()
        tr_loss += loss.item()
        
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        
        if step % 200==0:
            print("epoch: {}\t| cnt: {}\t|Loss: {}\t| precision: {:.4f}\t| time: {:.2f}".format(e, cnt, temp_loss/2000, cal_acc(label, pred), time.time()-start))
            temp_loss = 0
            start = time.time()
            
        if (step + 1) % global_params['gradient_accumulation_steps'] == 0:
            optim.step()
            optim.zero_grad()
            

    print("** ** * Saving fine - tuned model ** ** * ")
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    create_folder(file_config['model_path'])
    output_model_file = os.path.join(file_config['model_path'], file_config['model_name'])

    torch.save(model_to_save.state_dict(), output_model_file)
        
    cost = time.time() - start
    return tr_loss, cost

f = open(os.path.join(file_config['model_path'], file_config['file_name']), "w")
f.write('{}\t{}\t{}\n'.format('epoch', 'loss', 'time'))
for e in range(50):
    loss, time_cost = train(e, trainload)
    # loss = loss/data_len
    f.write('{}\t{}\t{}\n'.format(e, loss, time_cost))
f.close()    