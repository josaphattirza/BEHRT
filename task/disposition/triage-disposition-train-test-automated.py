import pickle
from pyspark.sql.functions import udf
from pyspark.sql import SparkSession
from pyspark.sql.functions import array, slice, size, expr
from pyspark.sql.types import StringType, ArrayType


import sys 
import os

sys.path.append('/home/josaphat/Desktop/research/ED-BERT-demo')

# Create a SparkSession

spark = SparkSession.builder \
    .appName("example_app") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.master", "local[*]") \
    .config("spark.executor.cores", "4") \
    .config("spark.default.parallelism", "16") \
    .getOrCreate()


# Load the Parquet file
df = spark.read.parquet("automated_los_final")

# Changing names to meet NextVisit needs in the training task
df = df.withColumnRenamed('icd_code', 'code')
df = df.withColumnRenamed('age_on_admittance', 'age')
df = df.withColumnRenamed('subject_id', 'patid')
df = df.withColumnRenamed('disposition', 'label')


# THE DIFFERENCE, SINCE WE ONLY WANT THE FINAL DISPOSITION AS LABEL
def get_label(arr):
    sep_count = arr.count('SEP')
    if sep_count == 1:
        return [arr[0]]  # return the first element in an array
    elif sep_count > 1:
        # find the index of the second last 'SEP'
        second_last_sep_index = len(arr) - arr[::-1].index('SEP') - arr[::-1][1:].index('SEP') - 2
        return [arr[second_last_sep_index + 1]]  # return the element after the second last 'SEP' in an array

get_label_udf = udf(get_label, ArrayType(StringType()))

# apply the UDF to the 'label' column
df = df.withColumn('label', get_label_udf(df['label']))


df.select("label").show()


def replace_unk_with_pad(df):
    replace_unk = udf(lambda arr: ['PAD' if item == 'UNK' else item for item in arr], ArrayType(StringType()))

    for column in df.columns:
        if column != 'patid':
            df = df.withColumn(column, replace_unk(df[column]))

    return df

df = replace_unk_with_pad(df)


# Count the occurrences of ['Yes'] and ['No'] in the 'label' column
label_counts = df.groupBy('label').count()

# Show the result
label_counts.show()



# Define a UDF to change 'EXPIRED', 'OTHER', 'HOME' to 'No', and 'ADMITTED' to 'Yes'
def map_labels(label):
    if label[0] == 'ADMITTED':
        return ['Yes']
    elif label[0] in ['EXPIRED', 'OTHER', 'HOME']:
        return ['No']
    else:
        return label

# Register the UDF
map_labels_udf = udf(map_labels, ArrayType(StringType()))

# Apply the UDF to the 'label' column
df = df.withColumn('label', map_labels_udf(df['label']))


# Count the occurrences of ['Yes'] and ['No'] in the 'label' column
label_counts = df.groupBy('label').count()

# Show the result
label_counts.show()


# Split the data into train and test sets
train_data, test_data = df.randomSplit([0.8, 0.2], seed=1234)

# Optionally, you can check the size of the train and test sets
print("Training set size:", train_data.count())
print("Test set size:", test_data.count())

train_data.write.parquet('automated_los_final_disposition_train_new_label')
test_data.write.parquet('automated_los_final_disposition_test_new_label')   




# # remove patients with more than one visit
# from pyspark.sql import functions as F
# # Define a user defined function (UDF) to calculate the count of 'SEP' in 'code' column
# count_sep = F.udf(lambda s: s.count('SEP'))
# # Apply the UDF to add a new column
# test_data = test_data.withColumn('length', count_sep(F.col('code')))
# train_data = train_data.withColumn('length', count_sep(F.col('code')))
# # Filter rows where 'length' equals to 1
# test_data = test_data.filter(F.col('length') == 1)
# train_data = train_data.filter(F.col('length') == 1)

# # Drop the 'length' column as it's not needed anymore
# # test_data = test_data.drop('length')


# print("Training set size:", train_data.count())
# print("Test set size:", test_data.count())



# train_data.write.parquet('automated_los_final_disposition_train_1visit')
# test_data.write.parquet('automated_los_final_disposition_test_1visit')