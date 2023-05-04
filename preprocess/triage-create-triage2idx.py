import pickle

triage2idx = {}

indexCount = 0

reservedKey = ["SEP","CLS","MASK","UNK","PAD"]

for key in reservedKey:
    triage2idx[key] = indexCount
    indexCount += 1


with open('./triage_categories.pkl','rb') as f:
    triage_categories_list = pickle.load(f)

triage_set = set(triage_categories_list)
triage_set.remove('UNK')


print(triage_set)
print(len(triage_set))

for e in triage_set:
    triage2idx[e] = indexCount
    indexCount += 1

print("eof")

saved_dict = dict()
saved_dict['triage2idx'] = triage2idx

print(len(saved_dict['triage2idx']))

with open('./triage2idx.pkl','wb') as f:
    pickle.dump(saved_dict, f)