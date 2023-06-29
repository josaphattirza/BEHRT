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
df = spark.read.parquet("behrt_triage_disposition_med_month_based")

# Changing names to meet NextVisit needs in the training task
df = df.withColumnRenamed('icd_code', 'code')
df = df.withColumnRenamed('age_on_admittance', 'age')
df = df.withColumnRenamed('subject_id', 'patid')
df = df.withColumnRenamed('disposition', 'label')


# THE DIFFERENCE, SINCE WE ONLY WANT THE FINAL DISPOSITION AS LABEL
# define a UDF to extract the last element of an array and add "SEP"
# last_elem_with_sep = udf(lambda arr: [arr[-1], arr[-1]], ArrayType(StringType()))
last_elem_with_sep = udf(lambda arr: [arr[-1]], ArrayType(StringType()))


# # apply the UDF to the 'label' column
# # this should be on ?
# df = df.withColumn('label', last_elem_with_sep(df['label']))

# # Define a UDF to replace 'EXPIRED' and 'OTHER' with 'OTHER'
# def replace_labels(label_list):
#     if label_list[0] in ['EXPIRED','OTHER']:
#         return ['OTHER']
#     else:
#         return label_list

# # Register the UDF
# replace_labels_udf = udf(replace_labels, ArrayType(StringType()))

# # Apply the UDF to the 'label' column
# df = df.withColumn('label', replace_labels_udf(df['label']))

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

train_data.write.parquet('behrt_triage_disposition_med_month_based_train2')
test_data.write.parquet('behrt_triage_disposition_med_month_based_test2')   