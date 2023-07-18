
import pickle
from pyspark.sql.functions import udf
from pyspark.sql import SparkSession
from pyspark.sql.functions import array, slice, size, expr
from pyspark.sql.types import StringType, ArrayType


import sys 
import os

sys.path.append('/home/josaphat/Desktop/research/BEHRT')

#Create PySpark SparkSession
spark = SparkSession.builder \
    .master("local[1]") \
    .appName("SparkApp") \
    .config("spark.driver.memory", "64g") \
    .config("spark.executor.memory", "64g") \
    .config("spark.master", "local[*]") \
    .config("spark.executor.cores", "16") \
    .getOrCreate()

# Load the Parquet file
# df = spark.read.parquet("behrt_triage_revisit_disposition_med_month_based")
df = spark.read.parquet("automated_los_final")

# Changing names to meet NextVisit needs in the training task
df = df.withColumnRenamed('icd_code', 'code')
df = df.withColumnRenamed('age_on_admittance', 'age')
df = df.withColumnRenamed('subject_id', 'patid')

df.write.parquet('automated_los_final_namechanged')
