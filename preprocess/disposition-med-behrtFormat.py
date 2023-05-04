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

# GOAL: To ensure that code, med, age_on_admittance, and disposition all has the same length
# NOTE:
# initial length:
# code = age
# med = disposition
# NOTE 2:
# CASE 1 : both code and medicine has same amount of visit (same amount of SEP)
# CASE 2 : code has more visit (more SEP)
# CASE 3 : med has more visit (more SEP)
def handle_arrays(icd_code, medicine, age_on_admittance,disposition):
    code_sublists = []
    med_sublists = []
    age_sublists = []
    disposition_sublists = []
    temp = []
    temp_age = []
    temp_disposition = []



    code_final_result = []
    med_final_result = []
    age_final_result = []
    disposition_final_result = []


    for item, age in zip(icd_code,age_on_admittance):
        if item == "SEP":
            code_sublists.append(temp)
            age_sublists.append(temp_age)
            temp = []
            temp_age = []
        else:
            temp.append(item)
            temp_age.append(age)
    if temp:
        code_sublists.append(temp)
        age_sublists.append(temp_age)

    for item,disp in zip(medicine,disposition):
        if item == "SEP":
            med_sublists.append(temp)
            disposition_sublists.append(temp_disposition)
            temp = []
            temp_disposition = []
        else:
            temp.append(item)
            temp_disposition.append(disp)
    if temp:
        med_sublists.append(temp)
        disposition_sublists.append(temp_disposition)

    for a,b,c,d in zip(code_sublists,med_sublists, age_sublists, disposition_sublists):
        if len(a) > len(b):
            diff = len(a) - len(b)
            for _ in range(diff):
                b.append('UNK')
                # c.append(c[0])
                d.append(d[0])
        if len(b) > len(a):
            diff = len(b) - len(a)
            for _ in range(diff):
                a.append('UNK')
                c.append(c[0])
                # d.append(d[0])

                
    for sublist in code_sublists:
        code_final_result.extend(sublist)
        code_final_result.append('SEP')

    for sublist in med_sublists:
        med_final_result.extend(sublist)
        med_final_result.append('SEP')

    for sublist in age_sublists:
        age_final_result.extend(sublist)
        age_final_result.append(sublist[0])

    for sublist in disposition_sublists:
        disposition_final_result.extend(sublist)
        disposition_final_result.append(sublist[0])


    if len(code_final_result) > len(med_final_result):
        # print("CASE 2")
        for _ in range(len(code_final_result)-len(med_final_result)-1):
            med_final_result.append('UNK')
            disposition_final_result.append(disposition_final_result[-1])

        med_final_result.append('SEP')
        disposition_final_result.append(disposition_final_result[-1])


    elif len(med_final_result) > len(code_final_result):
        # print("CASE 3")
        for _ in range(len(med_final_result)-len(code_final_result)-1):
            code_final_result.append('UNK')
            age_final_result.append(age_final_result[-1])

        code_final_result.append('SEP')
        age_final_result.append(age_final_result[-1])

        
    else:  
        # print("CASE 1")
        pass

    # print(len(final_result1))
    # print(len(final_result2))
    # print(len(age_final_result))
    # print(len(disposition_final_result))
    # print("=======")


    if(len(code_final_result)==len(med_final_result)==len(age_final_result)==len(disposition_final_result)):
        # print("ALL HAS SAME LENGTH")
        pass
    else:
        print('NOT SAME LENGTH')
        print(icd_code)
        print(medicine)
        print(code_final_result)
        print(med_final_result)
        print(age_final_result)
        print(disposition_final_result)

    return code_final_result, med_final_result, age_final_result, disposition_final_result



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

# merge ED admission with ED pyxis on same subject_id and stay_id
df_main = df_main.merge(df_pyxis, how='outer', on=['subject_id','stay_id'])
df_main = df_main.rename(columns={'name':'med'})

# IMPORTANT : Because we merge using outer, fill NaN with UNK
# REASON : some patients don't receive meds in their visit, but have diagnosis
df_main['med'] = df_main['med'].fillna('UNK')


df_main2 = df_main[['subject_id','stay_id','intime','age_on_admittance','med','disposition']]


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


# transform dataframe into spark due to unavailable method on normal pandas
sparkDF=spark.createDataFrame(df_main2)
sparkDF = sparkDF.groupBy(['subject_id','intime']).agg(F.collect_list('age_on_admittance').alias('age_on_admittance'),F.collect_list('med').alias('med'), F.collect_list('disposition').alias('disposition'))

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
sparkDF = sparkDF.withColumn('disposition_temp', extract_age('disposition')).withColumn('disposition', F.concat(F.col('disposition'),F.array(F.col('disposition_temp')))).drop('disposition_temp')


print(sparkDF.head())

w = Window.partitionBy('subject_id').orderBy('intime')
# sort and merge ccs and age
sparkDF = sparkDF \
    .withColumn('med', F.collect_list('med').over(w)) \
    .withColumn('age_on_admittance', F.collect_list('age_on_admittance').over(w)) \
    .withColumn('disposition', F.collect_list('disposition').over(w)) \
    .groupBy('subject_id') \
    .agg(F.max('med').alias('med'), F.max('age_on_admittance').alias('age_on_admittance'), F.max('disposition').alias('disposition'))


print(sparkDF.head())

def flatten_array(list2d):
    merged = list(itertools.chain.from_iterable(list2d))

    return merged

df_main = sparkDF.toPandas()

df_main["med"] = df_main['med'].apply(flatten_array)
df_main["age_on_admittance"] = df_main['age_on_admittance'].apply(flatten_array)
df_main["disposition"] = df_main['disposition'].apply(flatten_array)

df_main = df_main.drop('age_on_admittance', axis=1)

print(df_main)

schema = StructType([
    StructField("subject_id", IntegerType(), True),
    StructField("med", ArrayType(StringType(), True), True),
    # StructField("age_on_admittance", ArrayType(StringType(), True), True),
    StructField("disposition", ArrayType(StringType(), True), True),

])

med_sparkDF=spark.createDataFrame(df_main, schema=schema)

diagnosis_sparkDF = spark.read.parquet('./behrt_format_mimic4ed_month_based')

df = med_sparkDF.join(diagnosis_sparkDF, on='subject_id')

# Define the UDF
udf_handle_arrays = udf(handle_arrays, returnType=ArrayType(ArrayType(StringType())))

df = df.select("subject_id", udf_handle_arrays("icd_code", 
                                               "med", 
                                               "age_on_admittance",
                                               "disposition",
                                               ).alias("result"))
df = df.select("subject_id", 
               df.result.getItem(0).alias("icd_code"), 
               df.result.getItem(1).alias("med"), 
               df.result.getItem(2).alias("age_on_admittance"),
               df.result.getItem(3).alias("disposition"),
               )

# diagnoses = EHR(diagnoses).array_flatten(config['col_name']).array_flatten('age')
# diagnoses.write.parquet(config['output'])

# print(sparkDF)
df.show()

df.write.parquet('behrt_disposition_med_month_based')
