import pickle
from pyspark.sql.functions import udf
from pyspark.sql import SparkSession
from pyspark.sql.functions import array, slice, size, expr
from pyspark.sql.types import StringType, ArrayType


import sys 
import os

sys.path.append('/home/josaphat/Desktop/research/ED-BERT-demo')

# Create a SparkSession
spark = SparkSession.builder.appName("Read Parquet File").getOrCreate()
# Load the Parquet file
df = spark.read.parquet("behrt_med_format_mimic4ed_month_based")

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
        label = arr[second_last_sep_index + 1:]
        # print(label)
        # print(type(label))
        # print(type(label[0]))
        return label
    else:
        return []


# TODO, Change label content
df = df.withColumn("label", extract_second_last_values(df.code))


# define a UDF to remove strings after second last "SEP"
def remove_after_second_last_sep(arr):
    if arr.count("SEP") >= 2:
        sep_positions = [i for i, x in enumerate(arr) if x == "SEP"]
        second_last_sep = sep_positions[-2]
        return arr[:second_last_sep+1]
    else:
        return []

# register the UDF
udf_remove_after_second_last_sep = udf(remove_after_second_last_sep, ArrayType(StringType()))

# apply the UDF to the "code" column in the DataFrame
df = df.withColumn("code", udf_remove_after_second_last_sep("code"))

# apply the UDF to the "code" column in the DataFrame
df = df.withColumn("med", udf_remove_after_second_last_sep("med"))

df = df.filter(~(size("label") == 0))
df.select("label").show()


# Split the data into train and test sets
train_data, test_data = df.randomSplit([0.8, 0.2], seed=1234)

# Optionally, you can check the size of the train and test sets
print("Training set size:", train_data.count())
print("Test set size:", test_data.count())

train_data.write.parquet('behrt_med_format_mimic4ed_month_based_train')
test_data.write.parquet('behrt_med_format_mimic4ed_month_based_test')   