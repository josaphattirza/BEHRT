import sys
sys.path.append('/home/josaphat/Desktop/research/ED-BERT-demo')

from datetime import date
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import SparkSession, Row
from pyspark.sql import Window
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType, DoubleType
from itertools import chain
import itertools
import joblib

from common.handle_columns import fix_sequence_length
from common.triage_categories_pyspark import *
from common.format_columns_NTUH import format_data
from common.handleICD import *
from pyspark.sql.functions import col, transform, split, when, size, array, lit
from pyspark.sql.functions import concat_ws, round, concat
from pyspark.sql.functions import regexp_replace

from pyspark.sql.functions import when, col, split, array_distinct, expr
from pyspark.ml.feature import QuantileDiscretizer


# # Create a spark session and set the parallelism
# spark = SparkSession.builder \
#     .appName("example_app") \
#     .config("spark.driver.memory", "64g") \
#     .config("spark.executor.memory", "64g") \
#     .config("spark.master", "local[*]") \
#     .config("spark.executor.cores", "16") \
#     .config("spark.default.parallelism", 32) \
#     .getOrCreate()

spark = SparkSession.builder \
    .appName("example_app") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.master", "local[*]") \
    .config("spark.executor.cores", "4") \
    .config("spark.default.parallelism", "16") \
    .getOrCreate()

# Enable Arrow-based columnar data transfers optimization
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")


# # Load the pickle file into a Pandas DataFrame
# data = pd.read_pickle('data_all.pkl')
# data = pd.read_pickle('data/NTUH/data_new_all.pkl')
data = pd.read_pickle('data/NTUH/data_new_1000.pkl')
print('reached here')


# Iterate over the columns to convert uint8 and uint16 to int
# So that pyspark's Arrow optimization can be used
for column in data.columns:
    if data[column].dtype == 'uint8' or data[column].dtype == 'uint16':
        # Convert 'uint8' columns to 'int'
        data[column] = data[column].astype(int)
print('reached here')


# Define the function to convert elements to arrays ttasLv2
# Since it originally contains either integer or array of integers
def to_array(element):
    if isinstance(element, int):
        return [element]
    else:
        return element
# Apply the transformation
data['ttasLv2'] = data['ttasLv2'].apply(to_array)
print('reached here')


df = spark.createDataFrame(data)
print('successfully convert pandas to spark')


base_columns = ['PERSONID2',
                'ACCOUNTIDSE2',
                'ACCOUNTSEQNO']
triage_columns = ['SYSTOLIC','DIASTOLIC',
                    # 'MAP',
                    'OXYGEN','BODYTEMPERATURE',
                    'PULSE',
                    'RESPIRATION',
                    'PAININDEX','TRIAGE']

demographic_columns = ['Age']

important_columns = ['ICD10','ttasLv2','LABLOINC','StayHour']

label_columns = ['HospitalAdmission','Readmission3D','LOS24', 'L2D']



administration_columns = ['CLINICSBY','ASSIGNAREA',
'CONSULTDEPTCODE',
'CLINICSFOR',
'ISJOBRELATED',
'ONTHEWAYJOB',]

administration_columns2 = ['EMS',
'HolidayAdmit',
'dayzone',
'E_last_year_counts',
'I_last_year_counts',
'HolidayAfterDischarge']



scan_columns1 = ["WBC","HB","PLT","K","Band","Seg", "CRE",]

scan_columns2 = ["Na","ALT","GLU","HCO3","Troponin_I","Troponin_T","PH",]

scan_columns3 = ["LacticAcid","Ca","Lipase","CRP","AST","RBC", "HCT",]

scan_columns4 = ["RDW_CV", "OccultBlood_urine", "Urobilinogen",
    "Nitrite", "pro_BNP", "MCV", "BloodCulture"]

scan_columns_main = scan_columns1 + scan_columns2 + scan_columns3 + scan_columns4


GCS_columns = ['GCSE','GCSE_C','GCSV','GCSV_A','GCSV_E','GCSV_T','GCSM']


indicator_columns = ['BMI', 'SEX','FEVER','BLOODSUGAR']



# Merge the three lists into a single list
selected_columns = base_columns + demographic_columns + \
                        triage_columns + \
                        label_columns + important_columns + \
                        administration_columns + administration_columns2 +\
                        scan_columns_main + \
                        GCS_columns + indicator_columns

df_selected = df.select(*selected_columns)



# To ensure different ACCOUNTSEQNO is different visit
df_selected = df_selected.withColumn("ACCOUNTIDSE2", concat(col("ACCOUNTIDSE2"), col("ACCOUNTSEQNO")))



df_selected = df_selected.withColumnRenamed('PERSONID2', 'subject_id')
df_selected = df_selected.withColumnRenamed('ACCOUNTIDSE2', 'stay_id')
df_selected = df_selected.withColumnRenamed('ICD10', 'icd_code')
df_selected = df_selected.withColumnRenamed('HospitalAdmission', 'disposition')
df_selected = df_selected.withColumnRenamed('Readmission3D', 'revisit72')
df_selected = df_selected.withColumnRenamed('LABLOINC', 'lab')


# # Preprocess ttasLv2, icd_code, and lab
# Cast ttasLv2 contents into string
df_selected = df_selected.withColumn(
    'ttasLv2', 
    transform(col('ttasLv2'), lambda x: x.cast('string'))
)
# Take the first 3 ICD10 code
df_selected = df_selected.withColumn(
    'icd_code',
    concat_ws(
        ',',
        transform(
            split(col('icd_code'), ','),
            lambda x: x.substr(1, 3)
        )
    )
)
# Fill Null values in icd_10, icd_code, ttasLv2 and LABLOINC
df_selected = df_selected.withColumn('ttasLv2', col('ttasLv2').cast('string'))
df_selected = df_selected.withColumn(
    'ttasLv2',
    when(
        (col('ttasLv2') == '[]') | (col('ttasLv2').isNull()), 'UNK'
    ).otherwise(
        regexp_replace(
            regexp_replace(col('ttasLv2'), "[\\[\\]]", ""), 
            "\\s*,\\s*", ","
        )
    )
)
df_selected = df_selected.fillna({'icd_code': 'UNK', 'lab': 'UNK'})
# # End of ttasLv2, icd_code, and lab preprocessing



# # Preprocess labels
df_selected = df_selected.withColumn(
    'disposition',
    when(col('disposition') == True, 'Yes').otherwise('No')
)
df_selected = df_selected.withColumn(
    'revisit72',
    when(col('revisit72') == True, 'Yes').otherwise('No')
)
# # End of labels preprocessing



# # Preprocess LABLOINC
df_selected = df_selected.withColumn(
    "lab",
    when(
        col("lab") != "UNK",
        expr("concat_ws(',', array_distinct(transform(split(lab, ','), x -> substring(x, 1, length(x)-3))))")
    ).otherwise(col("lab"))
)
# # End of LABLOINC prepocessing



# # Preprocess LOS label
# Calculate Prolonged LOS > 6 hours
from pyspark.sql import functions as F
df_selected = df_selected.withColumn('los', 
                                     F.when((F.col('StayHour') + F.col('L2D')) > 6, 'Yes')
                                     .otherwise('No'))
# # End of LOS label preprocess



# # Preprocess age
# Create a new 'age' column that is 'Age' in months
df_selected = df_selected.withColumn('age_in_months', round(df['Age'] * 12).cast('integer'))
# Now convert the 'age' column into string
df_selected = df_selected.withColumn('age_in_months', col('age_in_months').cast('string'))
# # End of age preprocess




triage_columns = ['SYSTOLIC','DIASTOLIC',
                    # 'MAP',
                    'OXYGEN','BODYTEMPERATURE',
                    'PULSE',
                    'RESPIRATION',
                    'PAININDEX','TRIAGE']

# # Preprocess triage
df_selected = categorize_temp(df_selected, 'BODYTEMPERATURE')
df_selected = categorize_hr(df_selected, 'PULSE')
df_selected = categorize_rr(df_selected, 'RESPIRATION')
df_selected = categorize_o2sat(df_selected, 'OXYGEN')
df_selected = categorize_sbp(df_selected, 'SYSTOLIC')
df_selected = categorize_dbp(df_selected, 'DIASTOLIC')

# To convert the 'pain' column to numeric in PySpark, you can use the cast() function.
# However, if the conversion fails, it will return null values. 
df_selected = df_selected.withColumn('PAININDEX', df_selected['PAININDEX'].cast("double"))
df_selected = categorize_pain(df_selected, 'PAININDEX')
# df_selected = categorize_acuity(df_selected, 'acuity') #????
df_selected = df_selected.withColumn('TRIAGE', 
                         when((isnan(df_selected['TRIAGE'])) | (df_selected['TRIAGE'] == 0), 'UNK')
                         .otherwise(concat(lit('Triage-'), df_selected['TRIAGE'].cast('string'))))

# Create a new column that combines the values from the triages
df_selected = df_selected.withColumn('COMBINED_TRIAGE', concat_ws(',', *triage_columns))
# Now, drop the original columns
df_selected = df_selected.drop(*triage_columns)
df_selected = df_selected.withColumnRenamed('COMBINED_TRIAGE', 'triage')
# # End of triage preprocessing



# Preprocess administration data
# Cast columns to string, add prefix, and then combine
for col_name in administration_columns:
    df_selected = df_selected.withColumn(col_name, F.concat(F.lit(col_name.lower() + "-"), F.col(col_name).cast("string")))

# Create a new column that combines the values from the triages
df_selected = df_selected.withColumn('COMBINED_ADMINISTRATION', concat_ws(',', *administration_columns))
# Now, drop the original columns
df_selected = df_selected.drop(*administration_columns)
df_selected = df_selected.withColumnRenamed('COMBINED_ADMINISTRATION', 'admin')
# End of administration data preprocessing



# Preprocess administration data 2
# Cast columns to string, add prefix, and then combine
for col_name in administration_columns2:
    df_selected = df_selected.withColumn(col_name, F.concat(F.lit(col_name.lower() + "-"), F.col(col_name).cast("string")))

# Create a new column that combines the values from the triages
df_selected = df_selected.withColumn('COMBINED_ADMINISTRATION2', concat_ws(',', *administration_columns2))
# Now, drop the original columns
df_selected = df_selected.drop(*administration_columns2)
df_selected = df_selected.withColumnRenamed('COMBINED_ADMINISTRATION2', 'admin_ext')
# End of administration data 2 preprocessing 
 



# Preprocess scan data
# Apply QuantileDiscretizer to the float values of scan data
for col_name in scan_columns_main:
    # Set up a temporary column name for the transformed column
    temp_col_name = col_name + "_temp"

    # Set up and fit the discretizer
    discretizer = QuantileDiscretizer(numBuckets=5, inputCol=col_name, outputCol=temp_col_name)
    df_selected = discretizer.fit(df_selected).transform(df_selected)

    # Drop the original column and rename the transformed column
    df_selected = df_selected.drop(col_name).withColumnRenamed(temp_col_name, col_name)
# Cast columns to string, add prefix, and then combine
for col_name in scan_columns_main:
    df_selected = df_selected.withColumn(col_name, F.concat(F.lit(col_name.lower() + "-"), F.col(col_name).cast("string")))

# Create a new column that combines the values from the scans
# And drop the original columns
df_selected = df_selected.withColumn('scan1', concat_ws(',', *scan_columns1))
df_selected = df_selected.drop(*scan_columns1)
df_selected = df_selected.withColumn('scan2', concat_ws(',', *scan_columns2))
df_selected = df_selected.drop(*scan_columns2)
df_selected = df_selected.withColumn('scan3', concat_ws(',', *scan_columns3))
df_selected = df_selected.drop(*scan_columns3)
df_selected = df_selected.withColumn('scan4', concat_ws(',', *scan_columns4))
df_selected = df_selected.drop(*scan_columns4)
# End of scan preprocessing 


# Preprocess indicator data
indicator_that_contains_float = ['BMI', 'BLOODSUGAR']
# Apply QuantileDiscretizer to the float values of indicator data THAT CONTAINS float
for col_name in indicator_that_contains_float:
    # Set up a temporary column name for the transformed column
    temp_col_name = col_name + "_temp"

    # Set up and fit the discretizer
    discretizer = QuantileDiscretizer(numBuckets=5, inputCol=col_name, outputCol=temp_col_name)
    df_selected = discretizer.fit(df_selected).transform(df_selected)

    # Drop the original column and rename the transformed column
    df_selected = df_selected.drop(col_name).withColumnRenamed(temp_col_name, col_name)
# Cast columns to string, add prefix, and then combine
for col_name in indicator_columns:
    df_selected = df_selected.withColumn(col_name, F.concat(F.lit(col_name.lower() + "-"), F.col(col_name).cast("string")))
# Create a new column that combines the values from the indicators
# And drop the original columns
df_selected = df_selected.withColumn('indicator', concat_ws(',', *indicator_columns))
df_selected = df_selected.drop(*indicator_columns)
# End of indicator processing




# Preprocess GCS data
# Cast columns to string, add prefix, and then combine
for col_name in GCS_columns:
    df_selected = df_selected.withColumn(col_name, F.concat(F.lit(col_name.lower() + "-"), F.col(col_name).cast("string")))

# Create a new column that combines the values from the triages
df_selected = df_selected.withColumn('COMBINED_GCS', concat_ws(',', *GCS_columns))
# Now, drop the original columns
df_selected = df_selected.drop(*GCS_columns)
df_selected = df_selected.withColumnRenamed('COMBINED_GCS', 'gcs')
# End of GCS data preprocessing



df_selected = df_selected.orderBy(col("subject_id"), col("Age"))
df_selected, vocab_dict_NTUH = format_data(df_selected, 
                                           columns=['age_in_months',"disposition",
                                                    "revisit72","los"], 
                                           csv_columns=['icd_code','triage',
                                                        'ttasLv2','lab',
                                                        'admin',
                                                        'admin_ext',
                                                        'scan1',
                                                        'scan2',
                                                        'scan3',
                                                        'scan4',
                                                        'indicator',
                                                        'gcs',
                                                        ])
print('format data completed')




# remove the suffix _bucket from each column name
for col_name in df_selected.columns:
    if col_name.endswith('_bucket'):
        new_col_name = col_name.replace('_bucket', '')
        df_selected = df_selected.withColumnRenamed(col_name, new_col_name)

df_selected = df_selected.withColumnRenamed('age_in_months', 'age')

# Define the UDF
udf_fix_sequence_length = udf(fix_sequence_length, returnType=ArrayType(ArrayType(StringType())))

column_names = ["age","triage","icd_code","ttasLv2",
                "los","lab",
                "disposition","revisit72", 
                'admin','admin_ext',
                'scan1',
                'scan2',
                'scan3',
                'scan4',
                'indicator',
                'gcs'
                ]
# Apply the UDF
df_selected = df_selected.withColumn("result", udf_fix_sequence_length(*column_names))

# Split the result into individual columns
for i, column_name in enumerate(column_names):
    df_selected = df_selected.withColumn(column_name, df_selected.result.getItem(i))

df_selected = df_selected.drop("result")
print('fix_sequence_length_completed')







# df_selected.write.parquet('automated_NTUH')
# df_selected.write.parquet('automated_NTUH_lab_stayhour')
# df_selected.write.parquet('automated_NTUH_10_features')
df_selected.write.parquet('automated_NTUH_final')




# df_selected = df_selected.limit(1000)

# # df_selected.write.parquet('automated_NTUH_lab_stayhour_1000')
# df_selected.write.parquet('automated_NTUH_10_features_1000')
# df_selected.write.parquet('automated_NTUH_final_1000')
