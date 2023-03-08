import pickle

token2idx = {}

indexCount = 0

reservedKey = ["SEP","CLS","MASK","UNK","PAD"]

for key in reservedKey:
    token2idx[key] = indexCount
    indexCount += 1


with open('./icd_9to10.pkl','rb') as f:
    dicA = pickle.load(f)

sets = []
for icd9,icd10 in zip(dicA['icd9'],dicA['icd10']):
    if type(icd10) != str:
        buffer = set()
        buffer.add(icd9)
        buffer.update(icd10)
        sets.append(buffer)
    
    else:
        buffer = set()
        buffer.add(icd9)
        buffer.add(icd10)
        sets.append(buffer)

for icd9,icd10 in zip(dicA['icd9'],dicA['icd10']):
    if type(icd10) != str:
        buffer = set()
        buffer.add(icd9)
        buffer.update(icd10)
        sets.append(buffer)
    
    else:
        buffer = set()
        buffer.add(icd9)
        buffer.add(icd10)
        sets.append(buffer)
    
print(sets)




for i in range(12410):
    for j in sets[i]:
        token2idx[j] = indexCount
        indexCount += 1


print('eof 1')

with open('./icd_10to9.pkl','rb') as f:
    dicB = pickle.load(f)


for icd10,icd9 in zip(dicB['icd10'],dicB['icd9']):
    if icd10 not in token2idx:
        token2idx[icd10] = indexCount
        indexCount += 1
        continue

    if type(icd9) != str:
        for e in icd9:
            token2idx[e] = token2idx[icd10]
    
    else:
        token2idx[icd9] = token2idx[icd10]

print('eof 2') 



saved_dict = dict()
saved_dict['token2idx'] = token2idx

with open('./token2idx.pkl','wb') as f:
    pickle.dump(saved_dict, f)

