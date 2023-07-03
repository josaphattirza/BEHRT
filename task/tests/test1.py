from pyspark.sql.functions import explode
from pyspark.sql import SparkSession


# Create a SparkSession
spark = SparkSession.builder \
    .appName("Read Parquet File") \
    .getOrCreate()

df = spark.read.parquet('./automated_disposition_train')


# Explode the 'triage' column to create a new row for each element
df_exploded = df.select(explode('triage').alias('triage_element'))

# Get the distinct values of the 'triage_element' column
unique_values = df_exploded.select('triage_element').distinct()

# Print the unique values
unique_values.show()

df = spark.read.parquet('./automated_disposition_train')


# Explode the 'triage' column to create a new row for each element
df_exploded = df.select(explode('triage').alias('triage_element'))

# Get the distinct values of the 'triage_element' column
unique_values2 = df_exploded.select('triage_element').distinct()

unique_values2.show()
