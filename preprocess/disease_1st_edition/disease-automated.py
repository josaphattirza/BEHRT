import sys
sys.path.append('/home/josaphat/Desktop/research/ED-BERT-demo')

from datetime import date
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType
from itertools import chain
import itertools

from common.handle_columns import fix_sequence_length
from common.triage_categories import *
from common.format_columns import format_data
from common.handleICD import *
from pyspark.sql.functions import col




#Create PySpark SparkSession
spark = SparkSession.builder \
    .master("local[1]") \
    .appName("SparkApp") \
    .config("spark.driver.memory", "64g") \
    .config("spark.executor.memory", "64g") \
    .config("spark.master", "local[*]") \
    .config("spark.executor.cores", "16") \
    .getOrCreate()



def calculate_age_on_current_admission(admission_date,anchor_time,anchor_age):
    age = admission_date.year - anchor_time.year - ((admission_date.month, admission_date.day) < (anchor_time.month, anchor_time.day)) + anchor_age
    return age  

def calculate_age_on_current_admission_month_based(admission_date,anchor_time,anchor_age):
    age = relativedelta(admission_date, anchor_time).months + anchor_age*12
    return age  

# MIMIC IV
df_adm = pd.read_csv('data/mimic-iv-2.1/hosp/admissions.csv')
df_pat = pd.read_csv('data/mimic-iv-2.1/hosp/patients.csv')

# MIMIC IV ED
df_edstays = pd.read_csv('data/mimic-iv-ed-2.0/2.0/ed/edstays.csv')
df_eddiagnosis = pd.read_csv('data/mimic-iv-ed-2.0/2.0/ed/diagnosis.csv')

# # # For testing purposes
# df_adm = df_adm.head(100)
# df_pat = df_pat.head(100)
# df_edstays = df_edstays.head(100)
# df_eddiagnosis = df_eddiagnosis.head(100)

# taking relevant columns from MIMIC-IV-ED
df_edstays = df_edstays[['subject_id','hadm_id','stay_id','intime','outtime','arrival_transport','disposition']]

# transform column values to processable datetime
df_adm.admittime = pd.to_datetime(df_adm.admittime, format = '%Y-%m-%d %H:%M:%S', errors='coerce')
df_adm.dischtime = pd.to_datetime(df_adm.dischtime, format = '%Y-%m-%d %H:%M:%S', errors='coerce')    
df_adm.deathtime = pd.to_datetime(df_adm.deathtime, format = '%Y-%m-%d %H:%M:%S', errors='coerce')
df_edstays.intime = pd.to_datetime(df_edstays.intime, format = '%Y-%m-%d %H:%M:%S', errors='coerce')
df_edstays.outtime = pd.to_datetime(df_edstays.outtime, format = '%Y-%m-%d %H:%M:%S', errors='coerce')
df_adm = df_adm.sort_values(['subject_id', 'admittime'])


df_edstays = df_edstays.merge(df_pat, how='inner', on='subject_id')

# taking relevant columns, save it as Main Dataframe
df_diagnosis = df_adm[['subject_id','hadm_id',
'admittime','dischtime','deathtime']]

# find the first time patient is admitted to the hospital, save it in anchor_time
anchor_time = df_diagnosis.groupby('subject_id')['admittime'].min().reset_index()
anchor_time = anchor_time.rename(columns={'admittime':'anchor_time'})
df_diagnosis = df_diagnosis.merge(anchor_time, how='left', on='subject_id')

# merge patient info from MIMIC-IV with MIMIC-IV-ED edstays on same ID
df_diagnosis = df_diagnosis.merge(df_edstays, how='outer', on = ['subject_id','hadm_id'])


## FIXED PART
## Fill all anchor_time if we can find anchor_time is available
## Sometimes patient can have previous visits in adm,
## but during their ed visit, they dont have hadm_id
## making us unable to trace patient's anchor_time

## Sort the DataFrame by subject_id and anchor_time
df_diagnosis.sort_values(['subject_id', 'anchor_time'], inplace=True)
# Forward fill NaN values in anchor_time column within each subject_id group
df_diagnosis['anchor_time'].fillna(method='ffill', inplace=True)
# Remove rows with NaN values in stay_id column, 
# Since we don't need rows not from ED stays anyway
df_diagnosis.dropna(subset=['stay_id'], inplace=True)


# calculate patient age during admission
### intime = date on incoming to ED
### anchor_time = date when anchor_age is given
df_diagnosis['age_on_admittance'] = df_diagnosis.apply(lambda x: calculate_age_on_current_admission_month_based(x['intime'],x['anchor_time'],x['anchor_age']), axis=1)

# merge ED admission with ED diagnosis on same patient_id and admission_id
df_diagnosis = df_diagnosis.merge(df_eddiagnosis, how='inner', on=['subject_id','stay_id'])
# Sort the DataFrame by subject_id and stay_id
df_diagnosis = df_diagnosis.sort_values(['subject_id', 'intime'])
df_diagnosis = df_diagnosis[['subject_id','stay_id','age_on_admittance','icd_code','icd_version','intime']]


# Convert ICD 9 to 10
df_diagnosis = icd_converter(df_diagnosis)
df_diagnosis.drop('icd_version', axis=1, inplace=True)


# transform dataframe into spark for faster processing
df_diagnosis=spark.createDataFrame(df_diagnosis)
# # remove columns here that doesn't have age_on_admission, 
# # this means that in df_pat, the patient age_on_admission is not available
df_diagnosis = df_diagnosis.dropna(subset=["age_on_admittance"])
df_diagnosis = df_diagnosis.withColumn("age_on_admittance", col("age_on_admittance").cast("string"))

# feed the dataframe into the automated preprocessing framework
diagnosis_columns_to_process = ["icd_code", "age_on_admittance",]
df_diagnosis, diagnosis_dictionaries = format_data(df_diagnosis, diagnosis_columns_to_process)
df_diagnosis = df_diagnosis.orderBy('subject_id')


print("The number of rows in df_diagnosis is:", df_diagnosis.count())
# df_diagnosis.show()

df_diagnosis.write.parquet('automated_diagnosis_icd10')

















