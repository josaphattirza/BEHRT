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
# add icd9 and icd10 to the sets, same categories will be assigned the same index later
for icd9,icd10 in zip(dicA['icd9'],dicA['icd10']):
    if type(icd10) != str:
        same_category_icd9and10 = set()
        same_category_icd9and10.add(icd9)
        same_category_icd9and10.update(icd10)
        sets.append(same_category_icd9and10)
    
    else:
        same_category_icd9and10 = set()
        same_category_icd9and10.add(icd9)
        same_category_icd9and10.add(icd10)
        sets.append(same_category_icd9and10)
    
# len of this sets is 12410
# print(sets)

# assign icd9 and 10 that are in the same categories with same index
for i in range(12410):
    for j in sets[i]:
        token2idx[j] = indexCount
        indexCount += 1


print('eof 1')

print(len(token2idx))
print(indexCount)

with open('./icd_10to9.pkl','rb') as f:
    dicB = pickle.load(f)


for icd10,icd9 in zip(dicB['icd10'],dicB['icd9']):
    # in case icd10 have not been seen before
    if icd10 not in token2idx:
        token2idx[icd10] = indexCount
        indexCount += 1
        continue
    
    # add icd9 and icd10 to the sets, same categories will be assigned the index that have been found
    if type(icd9) != str:
        for e in icd9:
            token2idx[e] = token2idx[icd10]
    
    else:
        token2idx[icd9] = token2idx[icd10]

print('eof 2') 

print(len(token2idx))
print(indexCount)

values = token2idx.values()
values = set(values)

saved_dict = dict()
saved_dict['token2idx'] = token2idx

with open('./token2idx-grouped.pkl','wb') as f:
    pickle.dump(saved_dict, f)

