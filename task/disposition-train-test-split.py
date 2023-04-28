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
df = spark.read.parquet("behrt_disposition_med_month_based")

# Changing names to meet NextVisit needs in the training task
df = df.withColumnRenamed('icd_code', 'code')
df = df.withColumnRenamed('age_on_admittance', 'age')
df = df.withColumnRenamed('subject_id', 'patid')
df = df.withColumnRenamed('disposition', 'label')

df.select("label").show()


# Split the data into train and test sets
train_data, test_data = df.randomSplit([0.8, 0.2], seed=1234)

# Optionally, you can check the size of the train and test sets
print("Training set size:", train_data.count())
print("Test set size:", test_data.count())

train_data.write.parquet('behrt_disposition_med_month_based_train')
test_data.write.parquet('behrt_disposition_med_month_based_test')   