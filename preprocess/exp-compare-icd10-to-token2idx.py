import pandas as pd
import pickle
import sys,os

sys.path.append('/home/josaphat/Desktop/research/ED-BERT-demo')

df = pd.read_parquet('behrt_format_mimic4ed')

with open('token2idx.pkl', 'rb') as f:
    token2idx = pickle.load(f)
    token2idx = token2idx['token2idx']

# print(token2idx)

icd10set = set()

for arr in df['icd_code']:
    for e in arr:
        icd10set.add(e)


# find which icd9 or 10 codes are not present in the original token2idx keys
token2idx_set = set(token2idx.keys())
setdiff = icd10set - token2idx_set

# print(len(setdiff))

current_idx = max(token2idx.values())

for e in setdiff:
    token2idx[e] = current_idx 
    current_idx += 1

setdiff = icd10set - set(token2idx.keys())
# print(len(setdiff))

print(token2idx['S72092A'])

saved_dict = dict()
saved_dict['token2idx'] = token2idx

with open('./token2idx-added.pkl','wb') as f:
    pickle.dump(saved_dict, f)





