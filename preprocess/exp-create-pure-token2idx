import pickle

token2idx = {}

indexCount = 0

reservedKey = ["SEP","CLS","MASK","UNK","PAD"]

for key in reservedKey:
    token2idx[key] = indexCount
    indexCount += 1


with open('./icd_9to10.pkl','rb') as f:
    dicA = pickle.load(f)

with open('./icd_10to9.pkl','rb') as f:
    dicB = pickle.load(f)

sets = set()


for icd10,icd9 in zip(dicB['icd10'],dicB['icd9']):
    if type(icd9) != str:
        for e in icd9:
            sets.add(e)
    
    else:
        sets.add(icd9)

for icd9,icd10 in zip(dicA['icd9'],dicA['icd10']):
    if type(icd10) != str:
        for e in icd10:
            sets.add(e)
    
    else:
        sets.add(icd10)

print(sets)

icd_list = list(sets)

for e in icd_list:
    token2idx[e] = indexCount
    indexCount += 1

print("eof")

saved_dict = dict()
saved_dict['token2idx'] = token2idx

with open('./token2idx.pkl','wb') as f:
    pickle.dump(saved_dict, f)