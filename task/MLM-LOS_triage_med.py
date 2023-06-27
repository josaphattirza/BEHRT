import sys 
import os
sys.path.append('/home/josaphat/Desktop/research/BEHRT')
from model.Disposition_triage_med import BertForMultiLabelPrediction



# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from common.common import create_folder
from common.pytorch import load_model
import pytorch_pretrained_bert as Bert
from model.utils import age_vocab
from common.common import load_obj
from dataLoader.MLM_triage_med import MLMLoader
from torch.utils.data import DataLoader
import pandas as pd
from model.MLM_triage_med import BertForMaskedLM
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
        self.triage_vocab_size = config.get('triage_vocab_size')  # OWN EMBEDDINGS


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
    'med_vocab' : 'med2idx', 
    'triage_vocab' : 'triage2idx',
    'data': './behrt_triage_disposition_med_month_based_train/',  # formated data 
    'model_path': 'triage-med-MLM', # where to save model
    'model_name': 'triage-med-MLM-minvisit3-monthbased', # model name
    'file_name': 'triage-med-MLM-minvisit3-monthbased-log',  # log path
}
create_folder(file_config['model_path'])

global_params = {
    'max_seq_len': 64,
    'max_age': 110,
    'month': 1,
    'age_symbol': None,
    'min_visit': 3, # original is 5
    'gradient_accumulation_steps': 1 # originally is 1
}

optim_param = {
    'lr': 3e-5, # original is 3e-5
    'warmup_proportion': 0.1,
    'weight_decay': 0.01
}

train_params = {
    'batch_size': 32, # original is 256
    'use_cuda': True,
    'max_len_seq': global_params['max_seq_len'],
    'device': 'cuda:0'
}


BertVocab = load_obj(file_config['vocab'])
med_BertVocab = load_obj(file_config['med_vocab'])
triage_BertVocab = load_obj(file_config['triage_vocab'])
ageVocab, _ = age_vocab(max_age=global_params['max_age'], mon=global_params['month'], symbol=global_params['age_symbol'])
# print(ageVocab)


# OWN LABEL VOCAB , since we want to predict disposition
labelKey = ["UNK","ADMITTED","OTHER","EXPIRED","HOME"]
labelVocab = {}
for i,x in enumerate(labelKey):
    labelVocab[x] = i


data = pd.read_parquet(file_config['data'])
# print(data)
# remove patients with visits less than min visit
data['length'] = data['code'].apply(lambda x: len([i for i in range(len(x)) if x[i] == 'SEP']))
data = data[data['length'] >= global_params['min_visit']]
data = data.reset_index(drop=True)

# print(data)
Dset = MLMLoader(dataframe = data, token2idx = BertVocab['token2idx'], age2idx = ageVocab, med_token2idx=med_BertVocab['med2idx'], triage_token2idx=triage_BertVocab['triage2idx'], 
                 max_len=train_params['max_len_seq'], code='code', age='age', med='med', triage = 'triage')

trainload = DataLoader(dataset=Dset, batch_size=train_params['batch_size'], shuffle=True, num_workers=3)



from dataLoader.Disposition_triage import NextVisit
Dset2 = NextVisit(token2idx=BertVocab['token2idx'], label2idx=labelVocab, age2idx=ageVocab, 
                 med_token2idx=med_BertVocab['med2idx'], triage2idx=triage_BertVocab['triage2idx'],
                 dataframe=data, max_len=train_params['max_len_seq'])
trainload2 = DataLoader(dataset=Dset2, batch_size=train_params['batch_size'], shuffle=True, num_workers=1)





model_config = {
    'vocab_size': len(BertVocab['token2idx'].keys()), # number of disease + symbols for word embedding
    'med_vocab_size': len(med_BertVocab['med2idx'].keys()), # OWN EMBEDDINGS
    'triage_vocab_size': len(triage_BertVocab['triage2idx'].keys()), # OWN EMBEDDINGS

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

conf = BertConfig(model_config)
model = BertForMaskedLM(conf)


model = model.to(train_params['device'])
optim = adam(params=list(model.named_parameters()), config=optim_param)


feature_dict = {
    'word':True,
    'med':True, # OWN EMBEDDINGS
    'triage':True, # OWN EMBEDDINGS
    'seg':True,
    'age':True,
    'position': True
}

model2 = BertForMultiLabelPrediction(conf, num_labels=len(labelVocab.keys()), feature_dict=feature_dict)
model2 = model2.to(train_params['device'])
optim2 = adam(params=list(model2.named_parameters()), config=optim_param)


def cal_acc(label, pred):
    logs = nn.LogSoftmax()
    label=label.cpu().numpy()
    ind = np.where(label!=-1)[0]
    truepred = pred.detach().cpu().numpy()
    truepred = truepred[ind]
    truelabel = label[ind]
    truepred = logs(torch.tensor(truepred).log_softmax(dim=1))
    outs = [np.argmax(pred_x) for pred_x in truepred.numpy()]
    precision = skm.precision_score(truelabel, outs, average='micro')
    return precision

# OWN CHANGES
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer(classes=list(labelVocab.values()))
mlb.fit([[each] for each in list(labelVocab.values())])


def train(e, loader, loader2):
    tr_loss = 0
    temp_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    cnt= 0
    start = time.time()

    for step, (batch, batch2) in enumerate(zip(loader, loader2)):
        cnt +=1
        batch = tuple(t.to(train_params['device']) for t in batch)

        # age_ids, input_ids, posi_ids, segment_ids, attMask, masked_label = batch
        age_ids, input_ids, posi_ids, segment_ids, attMask, masked_label, \
            med_input_ids, triage_input_ids = batch
        
        # loss, pred, label = model(input_ids = input_ids, age_ids = age_ids, seg_ids = segment_ids, posi_ids = posi_ids,attention_mask=attMask, masked_lm_labels=masked_label)

        loss, pred, label, = model(input_ids=input_ids, med_input_ids=med_input_ids, triage_input_ids = triage_input_ids,
                                age_ids=age_ids, seg_ids=segment_ids, 
                                posi_ids=posi_ids, attention_mask=attMask, 
                                masked_lm_labels=masked_label,
                                )

        # print('loss', loss)
        # print('pred', pred)
        # print('label', label)

        if global_params['gradient_accumulation_steps'] >1:
            loss = loss/global_params['gradient_accumulation_steps']
        loss.backward()
        
        # MAYBE DIVIDE THIS BY 2? SINCE NOW I HAVE 2 LOSS (?)
        temp_loss += loss.item()
        tr_loss += loss.item()
        
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1


        # OWN CHANGES
        age_ids, input_ids, posi_ids, segment_ids, attMask, targets, _ , med_input_ids, triage_input_ids = batch2
        targets = torch.tensor(mlb.transform(targets.numpy()), dtype=torch.float32)


        age_ids = age_ids.to(train_params['device'])
        med_input_ids = med_input_ids.to(train_params['device']) # OWN EMBEDDINGS
        triage_input_ids = triage_input_ids.to(train_params['device']) # OWN EMBEDDINGS

        input_ids = input_ids.to(train_params['device'])
        posi_ids = posi_ids.to(train_params['device'])
        segment_ids = segment_ids.to(train_params['device'])
        attMask = attMask.to(train_params['device'])
        targets = targets.to(train_params['device'])
        
        loss, logits = model2(input_ids = input_ids, 
                             age_ids = age_ids, 
                             seg_ids = segment_ids, 
                             posi_ids = posi_ids,
                             attention_mask=attMask, labels=targets,
                             med_input_ids = med_input_ids,
                             triage_input_ids = triage_input_ids)
        
        if global_params['gradient_accumulation_steps'] >1:
            loss = loss/global_params['gradient_accumulation_steps']
        loss.backward()

        # MAYBE DIVIDE THIS BY 2? SINCE NOW I HAVE 2 LOSS (?)
        temp_loss += loss.item()
        tr_loss += loss.item()
        # OWN CHANGES UP
        
        if step % 200==0:
            final_precision = (cal_acc(label,pred))
            print("epoch: {}\t| cnt: {}\t|Loss: {}\t| precision: {:.4f}\t| time: {:.2f}".format(e, cnt, temp_loss/2000, final_precision, time.time()-start))
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
for e in range(20):
    # OWN CHANGES
    loss, time_cost = train(e, trainload, trainload2)
    # loss = loss/data_len
    f.write('{}\t{}\t{}\n'.format(e, loss, time_cost))
f.close()    