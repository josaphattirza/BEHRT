import pandas as pd

# Define a function to categorize body temperatures
def categorize_temp(temp):
    if pd.isna(temp):
        return 'UNK'
    elif temp < 95:
        return 'Temp-Hypothermia'
    elif temp < 95.5:
        return 'Temp-Low'
    elif temp < 99.5:
        return 'Temp-Normal'
    elif temp < 100.9:
        return 'Temp-Low grade fever'
    elif temp < 103:
        return 'Temp-Fever'
    else:
        return 'Temp-High fever'


# Define a function to categorize heart rates
def categorize_hr(hr):
    if pd.isna(hr):
        return 'UNK'
    elif hr < 60:
        return 'Hr-Bradycardia'
    elif hr < 100:
        return 'Hr-Normal'
    elif hr < 120:
        return 'Hr-Mild tachycardia'
    elif hr < 150:
        return 'Hr-Moderate tachycardia'
    else:
        return 'Severe tachycardia'


# Define a function to categorize respiratory rates
def categorize_rr(rr):
    if pd.isna(rr):
        return 'UNK'
    elif rr < 12:
        return 'Rr-Bradypnea'
    elif rr < 20:
        return 'Rr-Normal'
    elif rr < 24:
        return 'Rr-Mild tachypnea'
    elif rr < 30:
        return 'Rr-Moderate tachypnea'
    else:
        return 'Rr-Severe tachypnea'


# Define function to categorize oxygen saturation values
def categorize_o2sat(o2sat):
    if pd.isna(o2sat):
        return 'UNK'
    elif o2sat <= 80:
        return 'O2sat-Severe hypoxemia'
    elif o2sat <= 90:
        return 'O2sat-Moderate hypoxemia'
    elif o2sat <= 94:
        return 'O2sat-Mild hypoxemia'
    elif o2sat <= 97:
        return 'O2sat-Normal'
    else:
        return 'O2sat-Normal to high'


# Define a function to categorize SBP
def categorize_sbp(sbp):
    if pd.isna(sbp):
        return 'UNK'
    elif sbp >= 180:
        return 'Sbp-Severe hypertension'
    elif sbp >= 160:
        return 'Sbp-Stage 2 hypertension'
    elif sbp >= 140:
        return 'Sbp-Stage 1 hypertension'
    elif sbp >= 120:
        return 'Sbp-Prehypertension'
    else:
        return 'Sbp-Normal blood pressure'


# Define a function to categorize DBP
def categorize_dbp(dbp):
    if pd.isna(dbp):
        return 'UNK'
    if dbp >= 100:
        return 'Dbp-Stage 2 hypertension'
    elif dbp >= 90:
        return 'Dbp-Stage 1 hypertension'
    elif dbp >= 80:
        return 'Dbp-Prehypertension'
    elif dbp >= 60:
        return 'Dbp-Normal'
    else:
        return 'Dbp-Low'


# Define a function to categorize pain levels
def categorize_pain(pl):
    if pd.isna(pl):
        return 'UNK'
    elif pl == 0:
        return 'Pain-No pain'
    elif pl <= 3:
        return 'Pain-Mild pain'
    elif pl <= 6:
        return 'Pain-Moderate pain'
    elif pl <= 9:
        return 'Pain-Severe pain'
    elif 10 <= pl <= 13:
        return 'Pain-Very severe pain'
    else:
        return 'UNK'


# Define a function to categorize acuity scores
def categorize_acuity(acuity):
    if pd.isna(acuity):
        return 'UNK'
    elif acuity == 1:
        return 'Acuity-1'
    elif acuity == 2:
        return 'Acuity-2'
    elif acuity == 3:
        return 'Acuity-3'
    elif acuity == 4:
        return 'Acuity-4'
    elif acuity == 5:
        return 'Acuity-5'
    else:
        return 'UNK'
