from icd.icd9 import load as load_icd9
from icd.icd10 import load as load_icd10


# Load the pretrained token-to-index mapping for ICD-9
icd9_token2idx = load_icd9()

# Load the pretrained token-to-index mapping for ICD-10
icd10_token2idx = load_icd10()