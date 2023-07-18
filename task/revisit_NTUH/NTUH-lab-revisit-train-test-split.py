import pickle
from pyspark.sql.functions import udf
from pyspark.sql import SparkSession
from pyspark.sql.functions import array, slice, size, expr, col, regexp_replace
from pyspark.sql.types import StringType, ArrayType


import sys 
import os

sys.path.append('/home/josaphat/Desktop/research/BEHRT')

# Create a SparkSession
spark = SparkSession.builder.appName("Read Parquet File").getOrCreate()
# Load the Parquet file
# df = spark.read.parquet("automated_NTUH_lab_stayhour")
df = spark.read.parquet("automated_NTUH_lab_stayhour")


# Changing names to meet NextVisit needs in the training task
df = df.withColumnRenamed('icd_code', 'code')
df = df.withColumnRenamed('age_on_admittance', 'age')
df = df.withColumnRenamed('subject_id', 'patid')
df = df.withColumnRenamed('revisit72', 'label')


# # THE DIFFERENCE, SINCE WE ONLY WANT THE FINAL DISPOSITION AS LABEL
# # define a UDF to extract the last element of an array and add "SEP"
# # last_elem_with_sep = udf(lambda arr: arr, ArrayType(StringType()))
# # THE DIFFERENCE, SINCE WE ONLY WANT THE FINAL DISPOSITION AS LABEL
# def get_label(arr):
#     sep_count = arr.count('SEP')
#     if sep_count == 1:
#         return [arr[0]]  # return the first element in an array
#     elif sep_count > 1:
#         # find the index of the second last 'SEP'
#         second_last_sep_index = len(arr) - arr[::-1].index('SEP') - arr[::-1][1:].index('SEP') - 2
#         return [arr[second_last_sep_index + 1]]  # return the element after the second last 'SEP' in an array

# get_label_udf = udf(get_label, ArrayType(StringType()))

# # apply the UDF to the 'label' column
# df = df.withColumn('label', get_label_udf(df['label']))

# df.select("label").show()


def process_array(arr):
    # Count 'SEP' occurrences
    sep_count = arr.count('SEP')

    # Check if 'SEP' occurs more than once
    if sep_count > 1:
        # Find the index of the second last 'SEP'
        index = len(arr) - 1 - arr[::-1].index('SEP', 1)
        
        # Remove all elements after the second last 'SEP'
        arr = arr[:index]

    # Check if the last element is 'SEP' and remove it
    if arr[-1] == 'SEP':
        arr = arr[:-1]

    return arr

# Define the UDF
process_array_udf = udf(process_array, ArrayType(StringType()))

# Apply the UDF to the necessary columns
for column in ['code', 'label', 'triage', 'age', 'ttasLv2', 'lab']:
    df = df.withColumn(column, process_array_udf(df[column]))

def get_last_or_post_sep(arr):
    # Return the element after the last 'SEP', if 'SEP' exists in the array
    if 'SEP' in arr:
        index = arr[::-1].index('SEP')
        return [arr[-index]]
    # If no SEP exist, take the first element in the arr
    else:
        return [arr[0]]

# Define the UDF
get_last_or_post_sep_udf = udf(get_last_or_post_sep, ArrayType(StringType()))

# Apply the UDF to the 'label' column
df = df.withColumn('label', get_last_or_post_sep_udf(df['label']))

df.select("label").show()


def replace_unk_with_pad(df):
    replace_unk = udf(lambda arr: ['PAD' if item == 'UNK' else item for item in arr], ArrayType(StringType()))

    for column in df.columns:
        if column != 'patid':
            df = df.withColumn(column, replace_unk(df[column]))

    return df

df = replace_unk_with_pad(df)



# Replace the first 'P' in 'patid' with '1'
df = df.withColumn('patid', regexp_replace(col('patid'), '^P', '1'))




# Split the data into train and test sets
train_data, test_data = df.randomSplit([0.8, 0.2], seed=1234)

# Optionally, you can check the size of the train and test sets
print("Training set size:", train_data.count())
print("Test set size:", test_data.count())

# train_data.write.parquet('automated_NTUH_lab_stayhour_train')
# test_data.write.parquet('automated_NTUH_lab_stayhour_test')   
train_data.write.parquet('automated_NTUH_lab_stayhour_train')
test_data.write.parquet('automated_NTUH_lab_stayhour_test')   