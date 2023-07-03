import pickle

# Load the pickle file
with open('vocab_dict.pkl', 'rb') as f:
    vocab_dict = pickle.load(f)

# Print the keys
for key in vocab_dict.keys():
    print(key)

print(vocab_dict['triage2idx'])

# Load the pickle file
with open('triage2idx.pkl', 'rb') as f:
    vocab_dict3 = pickle.load(f)

print(vocab_dict3['triage2idx'])