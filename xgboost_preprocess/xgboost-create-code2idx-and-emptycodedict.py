import pickle

import numpy as np

code2idx = {}

indexCount = 0

reservedKey = ["SEP","CLS","MASK","UNK","PAD"]

for key in reservedKey:
    code2idx[key] = indexCount
    indexCount += 1


with open('./icd_code_categories.pkl','rb') as f:
    icd_code_categories = pickle.load(f)


print(icd_code_categories)
print(len(icd_code_categories))

for e in icd_code_categories:
    code2idx[e] = indexCount
    indexCount += 1

print("eof")

saved_dict = dict()
saved_dict['code2idx'] = code2idx

print(len(saved_dict['code2idx']))

with open('./code2idx.pkl','wb') as f:
    pickle.dump(saved_dict, f)

keys_to_remove = ['SEP', 'CLS', 'MASK', 'UNK', 'PAD']



# Remove specific keys
saved_dict['code2idx'] = {key: value for key, value in saved_dict['code2idx'].items() 
                          if key not in keys_to_remove}

# Add prefix 'icd_' to remaining keys and set all values to 0
saved_dict['code2idx'] = {f'icd_{key}': 0 for key in saved_dict['code2idx']}

with open('./empty_code2idx.pkl','wb') as f:
    pickle.dump(saved_dict, f)