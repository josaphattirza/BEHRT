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
df = spark.read.parquet("automated_final")

# Changing names to meet NextVisit needs in the training task
df = df.withColumnRenamed('icd_code', 'code')
df = df.withColumnRenamed('age_on_admittance', 'age')
df = df.withColumnRenamed('subject_id', 'patid')
df = df.withColumnRenamed('disposition', 'label')


# THE DIFFERENCE, SINCE WE ONLY WANT THE FINAL DISPOSITION AS LABEL
# AND TRYING TO PREDICT EARLY DISPOSITION
# THIS LOGIC CAN BE FIXED TO TRY FOR EARLY DISPOSITION
def process_array(arr, for_label=False):
    sep_count = arr.count('SEP')
    if sep_count == 1:
        label = arr[0]
        arr = [arr[0]]
    else:
        second_last_sep_index = len(arr) - 2 - arr[::-1].index('SEP')
        label = arr[second_last_sep_index+1]
        if not for_label:
            arr = arr[:second_last_sep_index+2]
    return [label], arr

last_elem_with_sep_for_label = udf(lambda arr: process_array(arr, True)[0], ArrayType(StringType()))
last_elem_with_sep_for_other = udf(lambda arr: process_array(arr)[1], ArrayType(StringType()))

# apply the UDF to the 'label', 'med' and 'code' columns
df = df.withColumn('label', last_elem_with_sep_for_label(df['label']))
df = df.withColumn('med', last_elem_with_sep_for_other(df['med']))
df = df.withColumn('code', last_elem_with_sep_for_other(df['code']))







df.select("label").show()


def replace_unk_with_pad(df):
    replace_unk = udf(lambda arr: ['PAD' if item == 'UNK' else item for item in arr], ArrayType(StringType()))

    for column in df.columns:
        if column != 'patid':
            df = df.withColumn(column, replace_unk(df[column]))

    return df

df = replace_unk_with_pad(df)


# Split the data into train and test sets
train_data, test_data = df.randomSplit([0.8, 0.2], seed=1234)

# Optionally, you can check the size of the train and test sets
print("Training set size:", train_data.count())
print("Test set size:", test_data.count())

train_data.write.parquet('automated_disposition_train3')
test_data.write.parquet('automated_disposition_test3')   