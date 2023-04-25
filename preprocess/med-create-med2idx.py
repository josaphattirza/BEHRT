import pickle

med2idx = {}

indexCount = 0

reservedKey = ["SEP","CLS","MASK","UNK","PAD"]

for key in reservedKey:
    med2idx[key] = indexCount
    indexCount += 1


with open('./medicines.pkl','rb') as f:
    med_list = pickle.load(f)

med_set = set(med_list)
med_set.remove('UNK')

# print(med_set)
# print(len(med_set))

for e in med_set:
    med2idx[e] = indexCount
    indexCount += 1

print("eof")

saved_dict = dict()
saved_dict['med2idx'] = med2idx

print(len(saved_dict['med2idx']))

with open('./med2idx.pkl','wb') as f:
    pickle.dump(saved_dict, f)