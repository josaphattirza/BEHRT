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
df = spark.read.parquet("behrt_format_mimic4ed_month_based_train")

print(df)
