import sys
sys.path.append('/home/josaphat/Desktop/research/BEHRT')


from common.icd_formatting import format_icd
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

# Read a CSV file
df_vital = spark.read.csv("/home/josaphat/Desktop/research/mimic-iv-ed-2.0/2.0/ed/vitalsign.csv", inferSchema=True, header=True)
df_eddiagnosis = spark.read.csv("/home/josaphat/Desktop/research/mimic-iv-ed-2.0/2.0/ed/diagnosis.csv", inferSchema=True, header=True)
df_vital = df_vital.limit(1000)
df_eddiagnosis = df_eddiagnosis.limit(1000)
df_eddiagnosis = format_icd(df_eddiagnosis)


# # Perform the join
# df_merged = df_eddiagnosis.join(df_vital, on=['subject_id','stay_id'], how='inner') # you can change 'inner' to 'outer', 'left_outer', 'right_outer' etc as needed


vital_columns_to_process = ["heartrate", "resprate"]
eddiagnosis_columns_to_process = ["icd_code", "icd_title"]

# Format df separately
df_vital, vital_dictionaries = format_data(df_vital, vital_columns_to_process)
df_eddiagnosis, diagnosis_dictionaries = format_data(df_eddiagnosis, eddiagnosis_columns_to_process)


# Perform the join
df_merged = df_eddiagnosis.join(df_vital, on=['subject_id'], how='outer')


# Add the postfix to each string in the merged list
merged_columns_to_process = eddiagnosis_columns_to_process + vital_columns_to_process
merged_columns_to_process = [s + '_bucket' for s in merged_columns_to_process]

# Define the UDF
udf_fix_sequence_length = udf(fix_sequence_length, returnType=ArrayType(ArrayType(StringType())))
# Apply the UDF
df_merged = df_merged.withColumn("result", udf_fix_sequence_length(*merged_columns_to_process))

# Split the result into individual columns
for i, column_name in enumerate(merged_columns_to_process):
    df_merged = df_merged.withColumn(column_name, df_merged.result.getItem(i))

df_merged = df_merged.drop("result")



print('finished')