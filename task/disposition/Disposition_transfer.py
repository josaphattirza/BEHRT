import sys
from matplotlib import pyplot as plt 

sys.path.append('/home/200700josa/ED-BERT-demo')

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
from sklearn.metrics import roc_auc_score, roc_curve
from common.common import load_obj
from model.utils import age_vocab
from dataLoader.Disposition_triage import NextVisit
from model.Disposition_triage_med import BertForMultiLabelPrediction
import warnings
warnings.filterwarnings(action='ignore')

file_config = {
    'vocab':'vocab_dict',  # vocab token2idx idx2token
    'train': './automated_los_final_disposition_train/',
    'test': './automated_los_final_disposition_train/', 
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
    'output_dir': 'finetune-disposition-triage',  # output dir
    'best_name': 'finetune-disposition-triage-monthbased-best', # output model name
    'max_len_seq': 150, # originally is 100, ?
    'max_age': 110,
    'age_year': False,
    'age_symbol': None,
    'min_visit': 3 # originally is 5
}


# pretrain_model_path = 'triage-med-MLM-automated/MLM-automated'  # pretrained MLM path
# pretrain_model_path = 'MLM-los-20epoch/MLM-los-automated'  # pretrained MLM path
# pretrain_model_path = 'MLM-los-20epoch-weighted/MLM-los-automated'  # pretrained MLM path
pretrain_model_path = 'NTUH-MLM-transfer/NTUH-MLM-transfer'  # pretrained MLM path



BertVocab = load_obj(file_config['vocab'])
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


# # OWN LABEL VOCAB , since we want to predict disposition
# labelKey = ["ADMITTED","OTHER","EXPIRED","HOME"]
labelKey = ["ADMITTED","OTHER","HOME"]
labelVocab = {}
for i,x in enumerate(labelKey):
    labelVocab[x] = i


model_config = {
    'vocab_size': len(BertVocab['icd_code2idx'].keys()), # number of disease + symbols for word embedding
    'med_vocab_size': len(BertVocab['icd_code2idx'].keys()), # OWN EMBEDDINGS, unused
    'triage_vocab_size': len(BertVocab['triage2idx'].keys()), # OWN EMBEDDINGS



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
Dset = NextVisit(token2idx=BertVocab['icd_code2idx'], label2idx=labelVocab, age2idx=ageVocab, 
                 med_token2idx=BertVocab['icd_code2idx'], triage2idx=BertVocab['triage2idx'],
                 dataframe=train, max_len=global_params['max_len_seq'])
trainload = DataLoader(dataset=Dset, batch_size=global_params['batch_size'], shuffle=True, num_workers=1)



test = pd.read_parquet(file_config['test'])
Dset = NextVisit(token2idx=BertVocab['icd_code2idx'], label2idx=labelVocab, age2idx=ageVocab, 
                 med_token2idx=BertVocab['icd_code2idx'], triage2idx=BertVocab['triage2idx'],
                 dataframe=test, max_len=global_params['max_len_seq'])
testload = DataLoader(dataset=Dset, batch_size=global_params['batch_size'], shuffle=False, num_workers=1)



# del model
conf = BertConfig(model_config)




# OWN CHANGES, comment all of this if want to use the old model
# Assuming `train` is a pandas DataFrame with the label column named "label"
class_labels = ["ADMITTED", "OTHER", "HOME"]  # List of class labels in order
# class_labels = ["ADMITTED","OTHER","EXPIRED","HOME"]

# Extract the single label value from each array
train['label'] = train['label'].apply(lambda x: x[0])

train['label'] = pd.Categorical(train['label'], categories=class_labels)
class_counts = train['label'].value_counts().reindex(class_labels, fill_value=0)
total_samples = len(train)
class_weights = total_samples / (len(class_labels) * class_counts)

print("Class Weights:", class_weights)
# model = BertForMultiLabelPrediction(conf, num_labels=len(labelVocab.keys()), feature_dict=feature_dict, weights=class_weights)
# COMMENT UNTIL HERE

# model = BertForMultiLabelPrediction(conf, num_labels=len(labelVocab.keys()), feature_dict=feature_dict)



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

# # in the original version, it's mode, is it correct?
# # OWN CHANGES
# model = load_model(pretrain_model_path, model)


# model = model.to(global_params['device'])
# optim = optimiser.adam(params=list(model.named_parameters()), config=optim_config)


import sklearn
def precision(logits, label):
    sig = nn.Sigmoid()
    output=sig(logits)
    label, output=label.cpu(), output.detach().cpu()
    tempprc= sklearn.metrics.average_precision_score(label.numpy(),output.numpy(), average='samples')
    print('tempprc', tempprc)
    return tempprc, output, label


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle


from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

def precision_test(logits, label, epoch, path):
    global max_auc

    softmax = nn.Softmax(dim=1)
    output = softmax(logits)

    output = output.detach().numpy()
    label = label.detach().numpy()

    n_classes = label.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    auc_score = 0.0
    total_samples = 0

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(label[:, i], output[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        num_samples = np.sum(label[:, i])
        auc_score += roc_auc[i] * num_samples
        total_samples += num_samples

    roc_auc["weighted"] = auc_score / total_samples

    print(f'Weighted average AUC: {roc_auc["weighted"]}')

    # if roc_auc["weighted"] > max_auc:
    max_auc = roc_auc["weighted"]

    plt.figure()
    for i, color in zip(range(n_classes), cycle(['aqua', 'darkorange', 'cornflowerblue'])):
        plt.plot(fpr[i], tpr[i], color=color, 
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic to Multi-Class')
    plt.legend(loc="lower right")
    
    # Add the weighted average AUC to the plot
    plt.text(0.32, 0.7, 'Weighted AUC = {0:0.5f}'.format(roc_auc["weighted"]), 
                bbox=dict(facecolor='red', alpha=0.5))

    # Make sure the directories exist
    os.makedirs(path, exist_ok=True)
    # Save the figure directly in the repetition folder and name it with the epoch count
    plt.savefig(f'{path}/epoch_{epoch}_roc_curve.png')
    plt.close()

    tempprc = sklearn.metrics.average_precision_score(label, output, average='samples')
    return tempprc, roc_auc, output, label,



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


def evaluation(e):
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

    # # Convert logits and targets to binary labels
    # y_binary = (y.detach().numpy() > 0.5).astype(int)
    # y_label_binary = y_label.detach().numpy()

    # # Apply argmax to obtain single class prediction
    # y_pred = np.argmax(y_binary, axis=1)

    # # Convert to one-hot encoding
    # y_pred_one_hot = np.eye(5)[y_pred]

    # train_labels = [1, 2, 3, 4]
    # mlb.fit(train_labels)

    # # Convert binary labels to human-readable form
    # logits_labels = mlb.inverse_transform(y_pred_one_hot)
    # targets_labels = mlb.inverse_transform(y_label_binary)

    # # Print out logits and targets
    # for i in range(len(logits_labels)):
    #     print(f"Input {i}:")
    #     print(f"Logits: {logits_labels[i]}")
    #     print(f"Targets: {targets_labels[i]}\n")

    # aps, roc, output, label = precision_test(y, y_label, e, "disposition-no-los")
    # aps, roc, output, label = precision_test(y, y_label, e, "disposition-los")
    aps, roc, output, label = precision_test(y, y_label, e, base_path)


    return aps, roc, tr_loss

    # if e == 10:
    #     aps, roc, output, label = precision_test(y, y_label)
    # return 0,0,0


feature_dict = {
    'word':True,
    'med':False, # OWN EMBEDDINGS
    'triage':True, # OWN EMBEDDINGS
    'seg':True,
    'age':True,
    'position': True
}



# Define the number of repetitions
n_repetitions = 5

# Loop over the number of repetitions
for r in range(n_repetitions):

    # Generate the features list for the base path
    features_list = []

    # Reset the best precision and max AUC for each repetition
    best_pre = 0.0 
    max_auc = 0.0  

    # Join the features with a hyphen
    features = "-".join(features_list)

    # Define the base path for this repetition
    base_path = f"result-MIMIC-IV/disposition-transfer/features-{features}/{r+1}"

    # Reset the model for each repetition (if needed)
    model = BertForMultiLabelPrediction(conf, num_labels=len(labelVocab.keys()), feature_dict=feature_dict, weights=class_weights)
    model = load_model(pretrain_model_path, model)
    model = model.to(global_params['device'])
    optim = optimiser.adam(params=list(model.named_parameters()), config=optim_config)

    # Loop over the epochs for each repetition
    for e in range(7):
        train(e)
        # Here you add the base_path to your precision_test function
        aps, roc, test_loss = evaluation(e, base_path)
        if aps > best_pre:
            best_pre = aps
        print('aps : {}'.format(aps))

    # Free up GPU memory
    torch.cuda.empty_cache()
