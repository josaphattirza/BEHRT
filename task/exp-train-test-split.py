import pickle
from pyspark.sql.functions import udf
from pyspark.sql import SparkSession
from pyspark.sql.functions import array, slice, size, expr
from pyspark.sql.types import StringType, ArrayType


import sys 
import os

sys.path.append('/home/josaphat/Desktop/research/BEHRT')

# Create a SparkSession
spark = SparkSession.builder.appName("Read Parquet File").getOrCreate()
# Load the Parquet file
df = spark.read.parquet("behrt_format_mimic4ed_month_based")

# Changing names to meet NextVisit needs in the training task
df = df.withColumnRenamed('icd_code', 'code')
df = df.withColumnRenamed('age_on_admittance', 'age')
df = df.withColumnRenamed('subject_id', 'patid')

@udf(returnType=ArrayType(StringType()))
def extract_second_last_values(arr):
    # find the second last index of "SEP" in the array
    sep_indices = [i for i, x in enumerate(arr) if x == "SEP"]
    if len(sep_indices) > 1:
        second_last_sep_index = sep_indices[-2]
        # extract all values after the second last "SEP"
        return arr[second_last_sep_index + 1:]
    else:
        return []


# TODO, Change label content
df = df.withColumn("label", extract_second_last_values(df.code))


# Split the data into train and test sets
train_data, test_data = df.randomSplit([0.8, 0.2], seed=1234)

# Optionally, you can check the size of the train and test sets
print("Training set size:", train_data.count())
print("Test set size:", test_data.count())

train_data.write.parquet('behrt_format_mimic4ed_month_based_train')
test_data.write.parquet('behrt_format_mimic4ed_month_based_test')