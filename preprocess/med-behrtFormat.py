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

from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType


#Create PySpark SparkSession
spark = SparkSession.builder \
    .master("local[1]") \
    .appName("SparkApp") \
    .getOrCreate()


def calculate_age_on_current_admission_month_based(admission_date,anchor_time,anchor_age):
    age = relativedelta(admission_date, anchor_time).months + anchor_age*12
    return age


def handle_arrays(array1, array2, age_on_admittance):
    sublists1 = []
    sublists2 = []
    age_sublists = []
    temp = []

    age_set = []
    age_set_counter = 0

    for item in age_on_admittance:
        if item not in age_set:
            age_set.append(item)


    final_result1 = []
    final_result2 = []
    age_final_result = []


    for item in array1:
        if item == "SEP":
            sublists1.append(temp)
            temp = []
        else:
            temp.append(item)

    if temp:
        sublists1.append(temp)

    for item in array2:
        if item == "SEP":
            sublists2.append(temp)
            temp = []
        else:
            temp.append(item)
    if temp:
        sublists2.append(temp)

    if len(sublists1) == len(sublists2):
        for a,b in zip(sublists1,sublists2):
            if len(a) > len(b):
                diff = len(a) - len(b)
                for _ in range(diff):
                    b.append('UNK')
            if len(b) > len(a):
                diff = len(b) - len(a)
                for _ in range(diff):
                    a.append('UNK')
            
            age_temp = [age_set[min(age_set_counter,len(age_set)-1)] for _ in range(len(a))]
            age_set_counter += 1
            age_sublists.append(age_temp)

    elif len(sublists1) > len(sublists2):
        for _ in range(len(array1)-len(array2)):
            sublists2[-1].append('UNK')
        sublists2[-1].append('SEP')

        for i in sublists1:
            age_temp = [age_set[min(age_set_counter,len(age_set)-1)] for _ in range(len(i))]
            age_set_counter += 1
            age_sublists.append(age_temp)

    elif len(sublists2) > len(sublists1):
        for _ in range(len(array2)-len(array1)):
            sublists1[-1].append('UNK')
        sublists1[-1].append('SEP')

        for i in sublists2:
            age_temp = [age_set[min(age_set_counter,len(age_set)-1)] for _ in range(len(i))]
            age_set_counter += 1
            age_sublists.append(age_temp)

                
    for sublist in sublists1:
        final_result1.extend(sublist)
        final_result1.append('SEP')

    for sublist in sublists2:
        final_result2.extend(sublist)
        final_result2.append('SEP')

    for sublist in age_sublists:
        age_final_result.extend(sublist)
        # age_final_result.append(age_set[min(age_set_counter,len(age_set)-1)])

        # TRY TO RUN THIS LINE
        age_final_result.append(sublist[-1])


    while(len(age_final_result) < len(final_result1)):
        age_final_result.append(age_set[min(age_set_counter,len(age_set)-1)])


    if(len(final_result1)!=len(final_result1)!=len(age_final_result)):
        print('NOT SAME LENGTH')
        print(array1)
        print(array2)
        print(final_result1)
        print(final_result2)

    return final_result1, final_result2, age_final_result



# MIMIC IV
df_adm = pd.read_csv('/home/josaphat/Desktop/research/mimic-iv-2.1/hosp/admissions.csv')
df_pat = pd.read_csv('/home/josaphat/Desktop/research/mimic-iv-2.1/hosp/patients.csv')

# MIMIC IV ED
df_edstays = pd.read_csv('/home/josaphat/Desktop/research/mimic-iv-ed-2.0/2.0/ed/edstays.csv')
df_eddiagnosis = pd.read_csv('/home/josaphat/Desktop/research/mimic-iv-ed-2.0/2.0/ed/diagnosis.csv')

# load ED-PYXIS 
df_pyxis = pd.read_csv('/home/josaphat/Desktop/research/mimic-iv-ed-2.0/2.0/ed/pyxis.csv')

# # For testing purposes
# df_adm = df_adm.head(100)
# df_pat = df_pat.head(100)
# df_edstays = df_edstays.head(100)
# df_eddiagnosis = df_eddiagnosis.head(100)
# df_pyxis = df_pyxis.head(100)

# taking relevant columns from MIMIC-IV-ED
df_edstays = df_edstays[['subject_id','hadm_id','stay_id','intime','outtime','arrival_transport','disposition']]


# transform column values to processable datetime
df_adm.admittime = pd.to_datetime(df_adm.admittime, format = '%Y-%m-%d %H:%M:%S', errors='coerce')
df_adm.dischtime = pd.to_datetime(df_adm.dischtime, format = '%Y-%m-%d %H:%M:%S', errors='coerce')    
df_adm.deathtime = pd.to_datetime(df_adm.deathtime, format = '%Y-%m-%d %H:%M:%S', errors='coerce')
df_edstays.intime = pd.to_datetime(df_edstays.intime, format = '%Y-%m-%d %H:%M:%S', errors='coerce')
df_edstays.outtime = pd.to_datetime(df_edstays.outtime, format = '%Y-%m-%d %H:%M:%S', errors='coerce')
df_adm = df_adm.sort_values(['subject_id', 'admittime'])



# merge admission info with patient demographics info
df_adm = df_adm.merge(df_pat, how='inner', on='subject_id')

# taking relevant columns, save it as Main Dataframe
df_main = df_adm[['subject_id','hadm_id',
'admittime','dischtime','deathtime','anchor_age']]

# find the first time patient is admitted to the hospital, save it in anchor_time
anchor_time = df_main.groupby('subject_id')['admittime'].min().reset_index()
anchor_time = anchor_time.rename(columns={'admittime':'anchor_time'})
df_main = df_main.merge(anchor_time, how='left', on='subject_id')

# merge patient info from MIMIC-IV with MIMIC-IV-ED edstays on same ID
df_main = df_main.merge(df_edstays, how='inner', on = ['subject_id','hadm_id'])

# calculate patient age during admission
### admittime = date on admission
### anchor_time = date when anchor_age is given
df_main['age_on_admittance'] = df_main.apply(lambda x: calculate_age_on_current_admission_month_based(x['admittime'],x['anchor_time'],x['anchor_age']), axis=1)

# merge ED admission with ED diagnosis on same patient_id and admission_id
df_main = df_main.merge(df_pyxis, how='outer', on=['subject_id','stay_id'])
df_main = df_main.rename(columns={'name':'med'})

# IMPORTANT : Because we merge using outer, fill NaN with UNK
# REASON : some patients don't receive meds in their visit, but have diagnosis
df_main['med'] = df_main['med'].fillna('UNK')

df_main2 = df_main[['subject_id','intime','age_on_admittance','med']]


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


# transform dataframe into spark due to unavailable method on normal pandas
sparkDF=spark.createDataFrame(df_main2)
sparkDF = sparkDF.groupBy(['subject_id','intime']).agg(F.collect_list('age_on_admittance').alias('age_on_admittance'),F.collect_list('med').alias('med'))

# print(sparkDF.head())


df_main = sparkDF.toPandas()

def array_add_element(array, val):
    return array + [val]

# print(df_main.head())

df_main['med'] = df_main.apply(lambda row: array_add_element(row['med'], 'SEP'),axis = 1)

# print(sparkDF.head())

sparkDF=spark.createDataFrame(df_main)

# add extra age to fill the gap of sep
extract_age = F.udf(lambda x: x[0])
sparkDF = sparkDF.withColumn('age_temp', extract_age('age_on_admittance')).withColumn('age_on_admittance', F.concat(F.col('age_on_admittance'),F.array(F.col('age_temp')))).drop('age_temp')

print(sparkDF.head())

w = Window.partitionBy('subject_id').orderBy('intime')
# sort and merge ccs and age
sparkDF = sparkDF.withColumn('med', F.collect_list('med').over(w)).withColumn('age_on_admittance', F.collect_list('age_on_admittance').over(w)).groupBy('subject_id').agg(F.max('med').alias('med'), F.max('age_on_admittance').alias('age_on_admittance'))

print(sparkDF.head())

def flatten_array(list2d):
    merged = list(itertools.chain.from_iterable(list2d))

    return merged

df_main = sparkDF.toPandas()

df_main["med"] = df_main['med'].apply(flatten_array)
df_main["age_on_admittance"] = df_main['age_on_admittance'].apply(flatten_array)
df_main = df_main.drop('age_on_admittance', axis=1)

print(df_main)

schema = StructType([
    StructField("subject_id", IntegerType(), True),
    StructField("med", ArrayType(StringType(), True), True),
    # StructField("age_on_admittance", ArrayType(StringType(), True), True)
])

med_sparkDF=spark.createDataFrame(df_main, schema=schema)

diagnosis_sparkDF = spark.read.parquet('./behrt_format_mimic4ed_month_based')

df = med_sparkDF.join(diagnosis_sparkDF, on='subject_id')

# Define the UDF
udf_handle_arrays = udf(handle_arrays, returnType=ArrayType(ArrayType(StringType())))

df = df.select("subject_id", udf_handle_arrays("icd_code", "med", "age_on_admittance").alias("result"))
df = df.select("subject_id", df.result.getItem(0).alias("icd_code"), df.result.getItem(1).alias("med"), df.result.getItem(2).alias("age_on_admittance"))

# diagnoses = EHR(diagnoses).array_flatten(config['col_name']).array_flatten('age')
# diagnoses.write.parquet(config['output'])

# print(sparkDF)
df.show()

df.write.parquet('behrt_med_format_mimic4ed_month_based')
