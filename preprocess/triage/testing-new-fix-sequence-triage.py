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


import sys
sys.path.append('/path/to/mimic4ed_benchmark')


#Create PySpark SparkSession
spark = SparkSession.builder \
    .master("local[1]") \
    .appName("SparkApp").getOrCreate()
    # .config("spark.driver.memory", "64g") \
    # .config("spark.executor.memory", "64g") \
    # .config("spark.master", "local[*]") \
    # .config("spark.executor.cores", "16") \
    # .getOrCreate()


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

# For testing purposes
df_adm = df_adm.head(100)
df_pat = df_pat.head(100)
df_edstays = df_edstays.head(100)
df_eddiagnosis = df_eddiagnosis.head(100)
df_pyxis = df_pyxis.head(100)
df_triage = df_triage.head(100)

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

# Iterate over each group
for _, group in grouped:
    # Calculate the time difference between intime and previous outtime
    group['time_diff'] = group['intime'] - group['outtime'].shift(1)

    # Mark the rows where the time difference is less than or equal to 72 hours (3 days)
    revisit_mask = group['time_diff'] <= pd.Timedelta(hours=72)

    # Update the revisit_in_72H column for the marked rows as 'Yes'
    df_edstays.loc[group.index[1:], 'revisit72'] = revisit_mask.map({True: 'Yes', False: 'No'})


# # # FASTER METHOD, CAN BE USED IF NEED TO RERUN
# # # BUT VERIFY RESULT IS THE SAME AS SLOWER METHOD
# # # PRETTY SURE IT'S THE SAME
# # Reset the DataFrame index
# df_edstays.reset_index(drop=True, inplace=True)

# # Convert intime and outtime columns to datetime format
# df_edstays['intime'] = pd.to_datetime(df_edstays['intime'])
# df_edstays['outtime'] = pd.to_datetime(df_edstays['outtime'])

# # Sort the DataFrame by patientID and intime in ascending order
# df_edstays.sort_values(['subject_id', 'intime'], inplace=True)

# # Calculate the time difference between intime and previous outtime for each row
# df_edstays['time_diff'] = df_edstays.groupby('subject_id')['intime'].diff()

# # Mark the rows where the time difference is less than or equal to 72 hours (3 days)
# revisit_mask = df_edstays['time_diff'] <= pd.Timedelta(hours=72)

# # Update the revisit_in_72H column based on the revisit_mask
# df_edstays['revisit_in_72H'] = revisit_mask.map({True: 'Yes', False: 'No'})

# # Drop the time_diff column
# df_edstays.drop('time_diff', axis=1, inplace=True)

# # Reset the DataFrame index
# df_edstays.reset_index(drop=True, inplace=True)


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
'admittime','dischtime','deathtime',]]

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

# merge ED admission with ED pyxis on same subject_id and stay_id
df_main = df_main.merge(df_pyxis, how='outer', on=['subject_id','stay_id'])
# Sort the DataFrame by subject_id and stay_id
df_main = df_main.sort_values(['subject_id', 'intime'])
df_main = df_main.rename(columns={'name':'med'})


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
df_triage['triage'] = df_triage_subset.apply(lambda x: x.tolist() + ['SEP'], axis=1)


# IMPORTANT : Because we merge using outer, fill NaN with UNK
# REASON : some patients don't receive meds in their visit, but have diagnosis
df_main['med'] = df_main['med'].fillna('UNK')
df_main2 = df_main[['subject_id','stay_id','intime','age_on_admittance','med','disposition','revisit72']]


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


# Group the original 8 disposition types into 4 types
# admitted = admitted, transfer, left against medical advice
# other = other, left without being seen, eloped
# expired = expired
# home = home
df_main2['disposition'] = df_main2['disposition'].replace({
    'ADMITTED': 'ADMITTED', 
    'TRANSFER': 'ADMITTED', 
    'LEFT AGAINST MEDICAL ADVICE': 'ADMITTED', 
    'OTHER': 'OTHER', 
    'LEFT WITHOUT BEING SEEN': 'OTHER', 
    'ELOPED': 'OTHER', 
    'EXPIRED': 'EXPIRED', 
    'HOME': 'HOME'})
df_main2['disposition'] = df_main2['disposition'].fillna('UNK')





# convert the 'med' column to a set and save it to a pickle file
med_list = list(df_main2['med'])
pd.Series(med_list).to_pickle('medicines.pkl')

# # remove columns here that doesn't have age_on_admission, 
# # this means that in df_pat, the patient age_on_admission is not available
df_main2 = df_main2.dropna(subset=["age_on_admittance"])
# Convert to string so we can add SEP later
df_main2['age_on_admittance'] = df_main2['age_on_admittance'].astype(int).astype(str)


# transform dataframe into spark due to unavailable method on normal pandas
sparkDF=spark.createDataFrame(df_main2)
sparkDF = sparkDF.groupBy(['subject_id','intime','stay_id']).agg(F.collect_list('age_on_admittance').alias('age_on_admittance'),
                                                       F.collect_list('med').alias('med'), 
                                                       F.collect_list('disposition').alias('disposition'),
                                                       F.collect_list('revisit72').alias('revisit72'),
                                                       )

# print(sparkDF.head())

df_main = sparkDF.toPandas()

def array_add_element(array, val):
    return array + [val]

# print(df_main.head())

df_main['med'] = df_main.apply(lambda row: array_add_element(row['med'], 'SEP'),axis = 1)

# merge admission, diagnosis, and pyxis with triage data (maybe consider using outer)
df_main = df_main.merge(df_triage[['subject_id', 'stay_id', 'triage']], how='inner', on=['subject_id', 'stay_id'])
# save triage categories list to generate triage2idx
triage_categories_list = list(df_main['triage'])
pd.Series(triage_categories_list).to_pickle('triage_categories.pkl')

# print(sparkDF.head())

df_main['age_on_admittance'] = df_main.apply(lambda row: array_add_element(row['age_on_admittance'], 'SEP'),axis = 1)
df_main['revisit72'] = df_main.apply(lambda row: array_add_element(row['revisit72'], 'SEP'),axis = 1)
df_main['disposition'] = df_main.apply(lambda row: array_add_element(row['disposition'], 'SEP'),axis = 1)


sparkDF=spark.createDataFrame(df_main)

# # add extra age to fill the gap of sep
# extract_age = F.udf(lambda x: x[0])
# sparkDF = sparkDF.withColumn('age_temp', extract_age('age_on_admittance')) \
#     .withColumn('age_on_admittance', F.concat(F.col('age_on_admittance'),F.array(F.col('age_temp')))) \
#     .drop('age_temp')
# sparkDF = sparkDF.withColumn('disposition_temp', extract_age('disposition')) \
#     .withColumn('disposition', F.concat(F.col('disposition'),F.array(F.col('disposition_temp')))) \
#     .drop('disposition_temp')
# sparkDF = sparkDF.withColumn('revisit72_temp', extract_age('revisit72')) \
#     .withColumn('revisit72', F.concat(F.col('revisit72'),F.array(F.col('revisit72_temp')))) \
#     .drop('revisit72_temp')


# print(sparkDF.head())

w = Window.partitionBy('subject_id').orderBy('intime')
# sort and merge ccs and age
sparkDF = sparkDF \
    .withColumn('med', F.collect_list('med').over(w)) \
    .withColumn('age_on_admittance', F.collect_list('age_on_admittance').over(w)) \
    .withColumn('disposition', F.collect_list('disposition').over(w)) \
    .withColumn('revisit72', F.collect_list('revisit72').over(w)) \
    .withColumn('triage', F.collect_list('triage').over(w)) \
    .groupBy('subject_id') \
    .agg(F.max('med').alias('med'), 
         F.max('age_on_admittance').alias('age_on_admittance'), 
         F.max('disposition').alias('disposition'),
         F.max('revisit72').alias('revisit72'),
         F.max('triage').alias('triage'),
         )


# print(sparkDF.head())

def flatten_array(list2d):
    merged = list(itertools.chain.from_iterable(list2d))
    return merged

df_main = sparkDF.toPandas()

df_main["med"] = df_main['med'].apply(flatten_array)
df_main["age_on_admittance"] = df_main['age_on_admittance'].apply(flatten_array)
df_main["disposition"] = df_main['disposition'].apply(flatten_array)
df_main["revisit72"] = df_main['revisit72'].apply(flatten_array)
df_main["triage"] = df_main['triage'].apply(flatten_array)

# print(df_main)

schema = StructType([
    StructField("subject_id", IntegerType(), True),
    StructField("med", ArrayType(StringType(), True), True),
    StructField("age_on_admittance", ArrayType(StringType(), True), True),
    StructField("disposition", ArrayType(StringType(), True), True),
    StructField("revisit72", ArrayType(StringType(), True), True),
    StructField("triage", ArrayType(StringType(), True), True),


])

med_sparkDF=spark.createDataFrame(df_main, schema=schema)

diagnosis_sparkDF = spark.read.parquet('./behrt_diagnosis_fixed_format_mimic4ed_month_based')
diagnosis_sparkDF = diagnosis_sparkDF.drop('age_on_admittance')

df = med_sparkDF.join(diagnosis_sparkDF, on='subject_id')

# Define the UDF
udf_fix_sequence_length = udf(fix_sequence_length, returnType=ArrayType(ArrayType(StringType())))

# Apply the UDF
df = df.withColumn("result", udf_fix_sequence_length("icd_code", 
                                                     "med", 
                                                     "age_on_admittance",
                                                     "disposition",
                                                     "revisit72",
                                                     "triage"))
# Split the result into individual columns
df = df.select("subject_id", 
               df.result.getItem(0).alias("icd_code"), 
               df.result.getItem(1).alias("med"), 
               df.result.getItem(2).alias("age_on_admittance"),
               df.result.getItem(3).alias("disposition"),
               df.result.getItem(4).alias("revisit72"),
               df.result.getItem(5).alias("triage"))


# diagnoses = EHR(diagnoses).array_flatten(config['col_name']).array_flatten('age')
# diagnoses.write.parquet(config['output'])

# print(sparkDF)
df.show()

# # Save the DataFrame as a pickle file
# with open('df.pkl', 'wb') as f:
#     pickle.dump(df.toPandas(), f)

df.write.parquet('behrtTT_triage_revisit_disposition_med_month_based')
