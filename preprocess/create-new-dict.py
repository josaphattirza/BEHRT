import pickle

# Load vocab_dict_NTUH.pkl
with open("vocab_dict_NTUH.pkl", "rb") as f:
    vocab_dict_NTUH = pickle.load(f)

# Rename age_in_months2idx to age2idx
vocab_dict_NTUH['age2idx'] = vocab_dict_NTUH.pop('age_in_months2idx')



# Load vocab_dict.pkl
with open("vocab_dict.pkl", "rb") as f:
    vocab_dict = pickle.load(f)

# Rename age_in_months2idx to age2idx
vocab_dict['age2idx'] = vocab_dict.pop('age_on_admittance2idx')

# Initialize a new dictionary
new_dict = {}

# Define a function to merge two dictionaries
def merge_dicts(dict1, dict2, key):
    # Start with dict1
    combined_dict = dict1[key]
    # Add entries from dict2 that are not in dict1
    for k, v in dict2[key].items():
        if k not in combined_dict:
            # set the new index as the maximum index in the combined dictionary + 1
            combined_dict[k] = max(combined_dict.values()) + 1
    return combined_dict

# Merge icd_code2idx, age2idx and triage2idx
for key in ['icd_code2idx', 'age2idx', 'triage2idx']:
    new_dict[key] = merge_dicts(vocab_dict_NTUH, vocab_dict, key)

# Save the combined dictionary
with open("new_dict.pkl", "wb") as f:
    pickle.dump(new_dict, f)
