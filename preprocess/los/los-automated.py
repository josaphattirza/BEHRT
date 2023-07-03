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

from common.handle_columns import fix_sequence_length
from common.triage_categories import *
from pyspark.sql.functions import explode
from common.format_columns import format_data



import sys
sys.path.append('/path/to/mimic4ed_benchmark')


#Create PySpark SparkSession
spark = SparkSession.builder \
    .master("local[1]") \
    .appName("SparkApp") \
    .config("spark.driver.memory", "64g") \
    .config("spark.executor.memory", "64g") \
    .config("spark.master", "local[*]") \
    .config("spark.executor.cores", "16") \
    .getOrCreate()


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

# Convert intime and outtime columns to datetime format
df_edstays['intime'] = pd.to_datetime(df_edstays['intime'])
df_edstays['outtime'] = pd.to_datetime(df_edstays['outtime'])

# Sort the DataFrame by patientID and intime in ascending order
df_edstays.sort_values(['subject_id', 'intime'], inplace=True)

# Group the DataFrame by patientID
grouped = df_edstays.groupby('subject_id')

# Initialize the revisit_in_72H column with default value 'No'
df_edstays['revisit72'] = 'No'

# # SLOWER METHOD
# # Iterate over each group
# for _, group in grouped:
#     # Calculate the time difference between intime and previous outtime
#     group['time_diff'] = group['intime'] - group['outtime'].shift(1)

#     # Mark the rows where the time difference is less than or equal to 72 hours (3 days)
#     revisit_mask = group['time_diff'] <= pd.Timedelta(hours=72)

#     # Update the revisit_in_72H column for the marked rows as 'Yes'
#     df_edstays.loc[group.index[1:], 'revisit72'] = revisit_mask.map({True: 'Yes', False: 'No'})


# # FASTER METHOD FOR REVISIT72, CAN BE USED IF NEED TO RERUN
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

# Update the revisit72 column based on the revisit_mask
df_edstays['revisit72'] = revisit_mask.map({True: 'Yes', False: 'No'})

# Shift the revisit72 column one row down for each 'subject_id'
df_edstays['revisit72'] = df_edstays.groupby('subject_id')['revisit72'].shift(-1)

# Fill NaN values with 'No' in case the last visit of a subject does not result in a revisit
df_edstays['revisit72'] = df_edstays['revisit72'].fillna('No')

# Drop the time_diff column
df_edstays.drop('time_diff', axis=1, inplace=True)

# Reset the DataFrame index
df_edstays.reset_index(drop=True, inplace=True)





# calculate the difference in hours directly and convert to 'Yes' or 'No' in the 'los' column
df_edstays['los'] = ((df_edstays['outtime'] - df_edstays['intime']).dt.total_seconds() / 3600).apply(lambda x: 'Yes' if x > 6 else 'No')




# transform column values to processable datetime
df_adm.admittime = pd.to_datetime(df_adm.admittime, format = '%Y-%m-%d %H:%M:%S', errors='coerce')
df_adm.dischtime = pd.to_datetime(df_adm.dischtime, format = '%Y-%m-%d %H:%M:%S', errors='coerce')    
df_adm.deathtime = pd.to_datetime(df_adm.deathtime, format = '%Y-%m-%d %H:%M:%S', errors='coerce')
df_edstays.intime = pd.to_datetime(df_edstays.intime, format = '%Y-%m-%d %H:%M:%S', errors='coerce')
df_edstays.outtime = pd.to_datetime(df_edstays.outtime, format = '%Y-%m-%d %H:%M:%S', errors='coerce')
df_adm = df_adm.sort_values(['subject_id', 'admittime'])



df_edstays = df_edstays.merge(df_pat, how='inner', on='subject_id')

# taking relevant columns, save it as Main Dataframe
df_med = df_adm[['subject_id','hadm_id',
'admittime','dischtime','deathtime',]]

# find the first time patient is admitted to the hospital, save it in anchor_time
anchor_time = df_med.groupby('subject_id')['admittime'].min().reset_index()
anchor_time = anchor_time.rename(columns={'admittime':'anchor_time'})
df_med = df_med.merge(anchor_time, how='left', on='subject_id')

# merge patient info from MIMIC-IV with MIMIC-IV-ED edstays on same ID
df_med = df_med.merge(df_edstays, how='outer', on = ['subject_id','hadm_id'])


## FIXED PART
## Fill all anchor_time if we can find anchor_time is available
## Sometimes patient can have previous visits in adm,
## but during their ed visit, they dont have hadm_id
## making us unable to trace patient's anchor_time

## Sort the DataFrame by subject_id and anchor_time
df_med.sort_values(['subject_id', 'anchor_time'], inplace=True)
# Forward fill NaN values in anchor_time column within each subject_id group
df_med['anchor_time'].fillna(method='ffill', inplace=True)
# Remove rows with NaN values in stay_id column, 
# Since we don't need rows not from ED stays anyway
df_med.dropna(subset=['stay_id'], inplace=True)


# calculate patient age during admission
### intime = date on incoming to ED
### anchor_time = date when anchor_age is given
df_med['age_on_admittance'] = df_med.apply(lambda x: calculate_age_on_current_admission_month_based(x['intime'],x['anchor_time'],x['anchor_age']), axis=1)




# merge ED admission with ED pyxis on same subject_id and stay_id
df_med = df_med.merge(df_pyxis, how='outer', on=['subject_id','stay_id'])
# Sort the DataFrame by subject_id and stay_id
df_med = df_med.sort_values(['subject_id', 'intime'])
df_med = df_med.rename(columns={'name':'med'})

# IMPORTANT : Because we merge using outer, fill NaN with UNK
# REASON : some patients don't receive meds in their visit, but have diagnosis
df_med['med'] = df_med['med'].fillna('UNK')
df_med = df_med[['subject_id','stay_id','age_on_admittance','med','disposition','revisit72','intime','los']]


# clear out medicine quantity and additonal extra information that is not needed (so we can group medicine)
# remove medicine dosage, and special characters
pattern1 = r'\s?\(.*\)|\s?\d+\w*|[^a-zA-Z\s]'
df_med['med'] = df_med['med'].str.replace(pattern1, '', regex = True)
# removed specific keywords
pattern2 = r'\b(?i)tab\b|\b(?i)vial\b|\b(?i)syr\b|\b(?i)kit\b|\b(?i)cap\b|\b(?i)cup\b|\b(?i)bag\b'
df_med['med'] = df_med['med'].str.replace(pattern2, '', regex = True)
# remove white spaces
pattern3 = r'\s+'
df_med['med'] = df_med['med'].str.replace(pattern3, '', regex = True)


# Group the original 8 disposition types into 4 types
# admitted = admitted, transfer, left against medical advice
# other = other, left without being seen, eloped
# expired = expired
# home = home
df_med['disposition'] = df_med['disposition'].replace({
    'ADMITTED': 'ADMITTED', 
    'TRANSFER': 'ADMITTED', 
    'LEFT AGAINST MEDICAL ADVICE': 'ADMITTED', 
    'OTHER': 'OTHER', 
    'LEFT WITHOUT BEING SEEN': 'OTHER', 
    'ELOPED': 'OTHER', 
    'EXPIRED': 'EXPIRED', 
    'HOME': 'HOME'})
df_med['disposition'] = df_med['disposition'].fillna('UNK')

# # remove columns here that doesn't have age_on_admission, 
# # this means that in df_pat, the patient age_on_admission is not available
df_med = df_med.dropna(subset=["age_on_admittance"])
# Convert to string so we can add SEP later
df_med['age_on_admittance'] = df_med['age_on_admittance'].astype(int).astype(str)


# transform dataframe into a pyspark dataframe
df_med=spark.createDataFrame(df_med)
# feed the dataframe into the automated preprocessing framework
med_columns_to_process = ["age_on_admittance","med","disposition","revisit72","los"]
df_med, diagnosis_dictionaries = format_data(df_med, med_columns_to_process)
df_med = df_med.orderBy('subject_id')
# FINISHED PREPROCESSING MED HERE


# STARTED PREPROCESSING TRIAGE HERE
# clean triage data, transform float value into 5 classes for each column
df_triage = df_triage[['subject_id', 'stay_id', 'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'acuity']]
df_triage['temperature'] = df_triage['temperature'].apply(categorize_temp)
df_triage['heartrate'] = df_triage['heartrate'].apply(categorize_hr)
df_triage['resprate'] = df_triage['resprate'].apply(categorize_rr)
df_triage['o2sat'] = df_triage['o2sat'].apply(categorize_o2sat)
df_triage['sbp'] = df_triage['sbp'].apply(categorize_sbp)
df_triage['dbp'] = df_triage['dbp'].apply(categorize_dbp)
df_triage['pain'] = pd.to_numeric(df_triage['pain'], errors='coerce')
df_triage['pain'] = df_triage['pain'].apply(categorize_pain)
df_triage['acuity'] = df_triage['acuity'].apply(categorize_acuity)


# create a new column 'triage' that contains an array of all the column values
triage_cols = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'acuity']
df_triage_subset = df_triage[triage_cols]
df_triage['triage'] = df_triage_subset.apply(lambda x: x.tolist(), axis=1)
df_triage = df_triage[['subject_id', 'stay_id', 'triage']]


# transform triage dataframe into a pyspark dataframe
df_triage = spark.createDataFrame(df_triage)
# Explode the 'triage' column (each stay id will have 8 triage information)
df_triage = df_triage.select("subject_id", "stay_id", explode(df_triage.triage).alias("triage"))


# feed the dataframe into the automated preprocessing framework
triage_columns_to_process = ["triage"]
df_triage, diagnosis_dictionaries = format_data(df_triage, triage_columns_to_process)
df_triage = df_triage.orderBy('subject_id')
# FINISHED PREPROCESSING TRIAGE HERE


# merge medicine and triage data (maybe consider using outer)
df_med_triage_merged = df_med.join(df_triage, ["subject_id"], "inner")

diagnosis_sparkDF = spark.read.parquet('./automated_diagnosis_icd10')
diagnosis_sparkDF = diagnosis_sparkDF.drop('age_on_admittance_bucket')
df_final = df_med_triage_merged.join(diagnosis_sparkDF, on='subject_id')


# remove the suffix _bucket from each column name
for col_name in df_final.columns:
    if col_name.endswith('_bucket'):
        new_col_name = col_name.replace('_bucket', '')
        df_final = df_final.withColumnRenamed(col_name, new_col_name)



# Define the UDF
udf_fix_sequence_length = udf(fix_sequence_length, returnType=ArrayType(ArrayType(StringType())))

column_names = ["icd_code", "med", "age_on_admittance", "disposition", "revisit72", "triage","los"]
# Apply the UDF
df_final = df_final.withColumn("result", udf_fix_sequence_length(*column_names))

# Split the result into individual columns
for i, column_name in enumerate(column_names):
    df_final = df_final.withColumn(column_name, df_final.result.getItem(i))

df_final = df_final.drop("result")


print("The number of rows in df_merged is:", df_med_triage_merged.count())
print("The number of rows in df_final is:", df_final.count())

# df_merged.show()

# df_med_triage_merged.write.parquet('automated_med_triage')
df_final.write.parquet('automated_los_final')
