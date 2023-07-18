import pickle

# Load the pickle file
with open('vocab_dict_NTUH.pkl', 'rb') as f:
    vocab_dict_NTUH = pickle.load(f)

# Print the keys
for key in vocab_dict_NTUH.keys():
    print(key)

print(vocab_dict_NTUH['triage2idx'])


# Sort the dictionary by keys in alphabetical order
sorted_dict = dict(sorted(vocab_dict_NTUH['StayHour2idx'].items()))

# Print the sorted dictionary
for key, value in sorted_dict.items():
    print(key, value)


# # Load the pickle file
# with open('triage2idx.pkl', 'rb') as f:
#     vocab_dict3 = pickle.load(f)

# print(vocab_dict3['triage2idx'])