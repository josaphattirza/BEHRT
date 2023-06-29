import pickle

# Load the pickle file
with open('vocab_dict.pkl', 'rb') as f:
    vocab_dict = pickle.load(f)

# Print the keys
for key in vocab_dict.keys():
    print(key)

print(vocab_dict['med2idx'])