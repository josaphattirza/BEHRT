from torch.utils.data.dataset import Dataset
import numpy as np
from dataLoader.utils import seq_padding,position_idx,index_seg,random_mask
import torch


class MLMLoader(Dataset):
    def __init__(self, dataframe, token2idx, med_token2idx, age2idx, max_len, code='code', age='age', med='med'):
        self.vocab = token2idx
        self.max_len = max_len
        self.code = dataframe[code]
        self.age = dataframe[age]
        self.age2idx = age2idx

        self.med_vocab = med_token2idx
        self.med = dataframe[med]

    def __getitem__(self, index):
        """
        return: age, code, position, segmentation, mask, label
        """

        # extract data
        age = self.age[index][(-self.max_len+1):]
        code = self.code[index][(-self.max_len+1):]
        med = self.med[index][(-self.max_len+1):]

        # avoid data cut with first element to be 'SEP'
        if code[0] != 'SEP':
            code = np.append(np.array(['CLS']), code)
            med = np.append(np.array(['CLS']), med)
            age = np.append(np.array(age[0]), age)  
        else:
            code[0] = 'CLS'
            med[0] = 'CLS'

        # # OPTION 2 (?)
        # if med[0] != 'SEP':
        #     med = np.append(np.array(['CLS']), med)
        #     age = np.append(np.array(age[0]), age)  
        # else:
        #     med[0] = 'CLS'

        # mask 0:len(code) to 1, padding to be 0
        mask = np.ones(self.max_len)
        mask[len(code):] = 0

        med_mask = np.ones(self.max_len)
        med_mask[len(med):] = 0

        # pad age sequence and code sequence
        age = seq_padding(age, self.max_len, token2idx=self.age2idx)

        tokens, code, label = random_mask(code, self.vocab)
        med_tokens, med, med_label = random_mask(med, self.med_vocab)


        # get position code and segment code
        tokens = seq_padding(tokens, self.max_len)
        position = position_idx(tokens)
        segment = index_seg(tokens)

        med_tokens = seq_padding(med_tokens, self.max_len)
        med_position = position_idx(med_tokens)
        med_segment = index_seg(med_tokens)

        # pad code and label
        code = seq_padding(code, self.max_len, symbol=self.vocab['PAD'])
        label = seq_padding(label, self.max_len, symbol=-1)

        med = seq_padding(med, self.max_len, symbol=self.med_vocab['PAD'])
        med_label = seq_padding(med_label, self.max_len, symbol=-1)

        return torch.LongTensor(age), torch.LongTensor(code), torch.LongTensor(position), torch.LongTensor(segment), \
               torch.LongTensor(mask), torch.LongTensor(label), \
                torch.LongTensor(med),  torch.LongTensor(med_position), \
                torch.LongTensor(med_segment),  torch.LongTensor(med_mask),  torch.LongTensor(med_label),

    def __len__(self):
        return len(self.code) 