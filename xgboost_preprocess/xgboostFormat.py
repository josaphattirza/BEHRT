import sys
sys.path.append('/home/josaphat/Desktop/research/BEHRT')

from datetime import date
import pickle
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType
from itertools import chain
import itertools

from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType


sys.path.append('/home/josaphat/Desktop/research/BEHRT')

from mimic4ed_benchmark.Benchmark_scripts.medcodes.diagnoses.icd_conversion import convert_9to10
from common.handleICD import *

from triage_categories import *


# #Create PySpark SparkSession
# spark = SparkSession.builder \
#     .master("local[1]") \
#     .appName("SparkApp") \
#     .config("spark.driver.memory", "64g") \
#     .config("spark.executor.memory", "64g") \
#     .config("spark.master", "local[*]") \
#     .config("spark.executor.cores", "16") \
#     .getOrCreate()

# # Sequence of processing:
# # 1. Add ICD and process it into multihot encoding
# # 2. Add Med and process it into multihot encoding
# # 3. Add triage (already categorical data)
# # 4. Process Disposition into 4 classes
# # 5. Process Revisit 3D


def calculate_age_on_current_admission_month_based(admission_date,anchor_time,anchor_age):
    age = relativedelta(admission_date, anchor_time).months + anchor_age*12
    return age

# MIMIC IV
df_adm = pd.read_csv('/home/josaphat/Desktop/research/mimic-iv-2.1/hosp/admissions.csv')
df_pat = pd.read_csv('/home/josaphat/Desktop/research/mimic-iv-2.1/hosp/patients.csv')

# MIMIC IV ED basic stay info and diagnosis
df_edstays = pd.read_csv('/home/josaphat/Desktop/research/mimic-iv-ed-2.0/2.0/ed/edstays.csv')
df_eddiagnosis = pd.read_csv('/home/josaphat/Desktop/research/mimic-iv-ed-2.0/2.0/ed/diagnosis.csv')

# load Pyxis 
df_pyxis = pd.read_csv('/home/josaphat/Desktop/research/mimic-iv-ed-2.0/2.0/ed/pyxis.csv')

# load triage
df_triage = pd.read_csv('/home/josaphat/Desktop/research/mimic-iv-ed-2.0/2.0/ed/triage.csv')

# # For testing purposes
# df_adm = df_adm.head(100)
# df_pat = df_pat.head(100)
# df_edstays = df_edstays.head(100)
# df_eddiagnosis = df_eddiagnosis.head(100)
# df_pyxis = df_pyxis.head(100)
# df_triage = df_triage.head(100)

# taking relevant columns from MIMIC-IV-ED
df_edstays = df_edstays[['subject_id','hadm_id','stay_id','intime','outtime','arrival_transport','disposition']]

# # FASTER METHOD, CAN BE USED IF NEED TO RERUN
# # BUT VERIFY RESULT IS THE SAME AS SLOWER METHOD
# # PRETTY SURE IT'S THE SAME
# Reset the DataFrame index
df_edstays.reset_index(drop=True, inplace=True)

# Convert intime and outtime columns to datetime format
df_edstays['intime'] = pd.to_datetime(df_edstays['intime'])
df_edstays['outtime'] = pd.to_datetime(df_edstays['outtime'])

# Sort the DataFrame by patientID and intime in ascending order
df_edstays.sort_values(['subject_id', 'intime'], inplace=True)

# Calculate the time difference between intime and previous outtime for each row
df_edstays['time_diff'] = df_edstays.groupby('subject_id')['intime'].diff()

# Mark the rows where the time difference is less than or equal to 72 hours (3 days)
revisit_mask = df_edstays['time_diff'] <= pd.Timedelta(hours=72)

# Update the revisit_in_72H column based on the revisit_mask
df_edstays['revisit_in_72H'] = revisit_mask.map({True: np.bool_(True), False: np.bool_(False)})

# Drop the time_diff column
df_edstays.drop('time_diff', axis=1, inplace=True)

# Reset the DataFrame index
df_edstays.reset_index(drop=True, inplace=True)


# transform column values to processable datetime
df_adm.admittime = pd.to_datetime(df_adm.admittime, format = '%Y-%m-%d %H:%M:%S', errors='coerce')
df_adm.dischtime = pd.to_datetime(df_adm.dischtime, format = '%Y-%m-%d %H:%M:%S', errors='coerce')    
df_adm.deathtime = pd.to_datetime(df_adm.deathtime, format = '%Y-%m-%d %H:%M:%S', errors='coerce')
df_edstays.intime = pd.to_datetime(df_edstays.intime, format = '%Y-%m-%d %H:%M:%S', errors='coerce')
df_edstays.outtime = pd.to_datetime(df_edstays.outtime, format = '%Y-%m-%d %H:%M:%S', errors='coerce')
df_adm = df_adm.sort_values(['subject_id', 'admittime'])

df_edstays = df_edstays.merge(df_pat, how='inner', on='subject_id')


# taking relevant columns, save it as Main Dataframe
df_main = df_adm[['subject_id','hadm_id',
'admittime','dischtime','deathtime']]

# find the first time patient is admitted to the hospital, save it in anchor_time
anchor_time = df_main.groupby('subject_id')['admittime'].min().reset_index()
anchor_time = anchor_time.rename(columns={'admittime':'anchor_time'})
df_main = df_main.merge(anchor_time, how='left', on='subject_id')

# merge patient info from MIMIC-IV with MIMIC-IV-ED edstays on same ID
df_main = df_main.merge(df_edstays, how='outer', on = ['subject_id','hadm_id'])


## FIXED PART
## Fill all anchor_time if we can find anchor_time is available
## Sometimes patient can have previous visits in adm,
## but during their ed visit, they dont have hadm_id
## making us unable to trace patient's anchor_time

## Sort the DataFrame by subject_id and anchor_time
df_main.sort_values(['subject_id', 'anchor_time'], inplace=True)
# Forward fill NaN values in anchor_time column within each subject_id group
df_main['anchor_time'].fillna(method='ffill', inplace=True)
# Remove rows with NaN values in stay_id column, 
# Since we don't need rows not from ED stays anyway
df_main.dropna(subset=['stay_id'], inplace=True)


# calculate patient age during admission
### intime = date on incoming to ED
### anchor_time = date when anchor_age is given
df_main['age_on_admittance'] = df_main.apply(lambda x: calculate_age_on_current_admission_month_based(x['intime'],x['anchor_time'],x['anchor_age']), axis=1)

# merge ED admission with ED diagnosis on same patient_id and admission_id
df_main = df_main.merge(df_eddiagnosis, how='inner', on=['subject_id','stay_id'])
# Sort the DataFrame by subject_id and stay_id
df_main = df_main.sort_values(['subject_id', 'intime'])
df_main = df_main[['subject_id','stay_id',
                   'hadm_id','intime',
                   'outtime','age_on_admittance',
                   'disposition',
                   'revisit_in_72H',
                   'icd_code','icd_version']]

print('reached here 1')

# df_main = spark.createDataFrame(df_main)

# Call the commorbidity function with Dask DataFrame
df_icds = np_icd(df_main)
# df_main = commorbidity(df_main, mode='digit3')


# Drop the 'icd_code' and 'icd_version' columns
df_main = df_main.drop(columns=['icd_code', 'icd_version'])
# Drop duplicate rows based on 'subject_id' and 'stay_id' columns
df_main = df_main.drop_duplicates(subset=['subject_id', 'stay_id'])
# Reset the index if desired
df_main = df_main.reset_index(drop=True)


# merge ED admission with ED pyxis on same subject_id and stay_id
df_main = df_main.merge(df_pyxis, how='outer', on=['subject_id','stay_id'])
# Sort the DataFrame by subject_id and stay_id
df_main = df_main.sort_values(['subject_id', 'intime'])
df_main = df_main.rename(columns={'name':'med'})

# IMPORTANT : Because we merge using outer, fill NaN with UNK
# REASON : some patients don't receive meds in their visit, but have diagnosis
df_main['med'] = df_main['med'].fillna('UNK')
df_main2 = df_main[['subject_id','stay_id',
                    'intime','age_on_admittance',
                    'disposition',
                    'revisit_in_72H',
                    'med']]


# clear out medicine quantity and additonal extra information that is not needed (so we can group medicine)
# remove medicine dosage, and special characters
pattern1 = r'\s?\(.*\)|\s?\d+\w*|[^a-zA-Z\s]'
df_main2['med'] = df_main2['med'].str.replace(pattern1, '', regex = True)

# removed specific keywords
pattern2 = r'\b(?i)tab\b|\b(?i)vial\b|\b(?i)syr\b|\b(?i)kit\b|\b(?i)cap\b|\b(?i)cup\b|\b(?i)bag\b'
df_main2['med'] = df_main2['med'].str.replace(pattern2, '', regex = True)

# remove white spaces
pattern3 = r'\s+'
df_main2['med'] = df_main2['med'].str.replace(pattern3, '', regex = True)

# convert the 'med' column to a set and save it to a pickle file
med_list = list(df_main2['med'])
pd.Series(med_list).to_pickle('medicines.pkl')

# remove columns here that doesn't have age_on_admission, 
# this means that in df_pat, the patient age_on_admission is not available
df_main = df_main2.dropna(subset=["age_on_admittance"])



df_meds = np_med(df_main)

# Drop the 'icd_code' and 'icd_version' columns
df_main = df_main.drop(columns=['med'])
# Drop duplicate rows based on 'subject_id' and 'stay_id' columns
df_main = df_main.drop_duplicates(subset=['subject_id', 'stay_id'])
# Reset the index if desired
df_main = df_main.reset_index(drop=True)

df_main = pd.merge(df_main, df_icds, on='stay_id', how='left')

df_main = pd.merge(df_main, df_meds, on='stay_id', how='left')


# clean triage data, transform float value into 5 classes for each column
df_triage = df_triage[['subject_id', 'stay_id', 'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'acuity']]
# df_triage['temperature'] = df_triage['temperature'].apply(categorize_temp)
# df_triage['heartrate'] = df_triage['heartrate'].apply(categorize_hr)
# df_triage['resprate'] = df_triage['resprate'].apply(categorize_rr)
# df_triage['o2sat'] = df_triage['o2sat'].apply(categorize_o2sat)
# df_triage['sbp'] = df_triage['sbp'].apply(categorize_sbp)
# df_triage['dbp'] = df_triage['dbp'].apply(categorize_dbp)
df_triage['pain'] = pd.to_numeric(df_triage['pain'], errors='coerce')
# df_triage['pain'] = df_triage['pain'].apply(categorize_pain)
# df_triage['acuity'] = df_triage['acuity'].apply(categorize_acuity)

print()

df_main = pd.merge(df_main, df_triage, on=['subject_id','stay_id'], how='left')

print()



# Group the original 8 disposition types into 4 types
# admitted = admitted, transfer, left against medical advice
# other = other, left without being seen, eloped
# expired = expired
# home = home
df_main['disposition'] = df_main['disposition'].replace({
    'ADMITTED': np.bool_(True), 
    'TRANSFER': np.bool_(True), 
    'LEFT AGAINST MEDICAL ADVICE': np.bool_(True), 
    'OTHER': np.bool_(False), 
    'LEFT WITHOUT BEING SEEN': np.bool_(False), 
    'ELOPED': np.bool_(False), 
    'EXPIRED': np.bool_(False), 
    'HOME': np.bool_(False)})

print('reached here 5')


# # DICTIONARY CREATION
# # save icd code categories set to create code2idx
# icd_code_categories = set(df_main['icd_code'])
# with open('icd_code_categories.pkl', 'wb') as f:
#     pickle.dump(icd_code_categories, f)


with open('./df_main.pkl','wb') as f:
    pickle.dump(df_main, f)


# Create a new column that combines 'subject_id' and 'stay_id', separated by a comma
df_main['index'] = df_main['subject_id'].astype(str) + ',' + df_main['stay_id'].astype(str)
# Set the new column as the index
df_main.set_index('index', inplace=True)
df_main = df_main.drop(columns=['subject_id', 'stay_id', 'intime'])

df_main = df_main.rename(columns={'disposition': 'HospitalAdmission',
                                   'revisit_in_72H':'Readmission'})
data = {}
# Create 'Y' dataframe with index and 'disposition'
data['Y'] = df_main[['HospitalAdmission', 'Readmission']]
# Create 'X' dataframe with all other columns
data['X'] = df_main.drop(columns=['HospitalAdmission','Readmission'])




# import joblib

# # Save the dictionary as a joblib file
# joblib.dump(data, 'data.joblib')

with open('./data.pkl','wb') as f:
    pickle.dump(data, f)



print()