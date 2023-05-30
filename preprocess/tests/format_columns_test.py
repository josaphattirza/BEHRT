import sys
sys.path.append('/home/josaphat/Desktop/research/BEHRT')

from datetime import date
import pickle
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType
from itertools import chain
import itertools

from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

from common.handle_columns import fix_sequence_length
from common.triage_categories import *
from common.format_columns import format_numerical

import sys
sys.path.append('/path/to/mimic4ed_benchmark')


#Create PySpark SparkSession
spark = SparkSession.builder \
    .master("local[1]") \
    .appName("SparkApp").getOrCreate()
    # .config("spark.driver.memory", "64g") \
    # .config("spark.executor.memory", "64g") \
    # .config("spark.master", "local[*]") \
    # .config("spark.executor.cores", "16") \
    # .getOrCreate()

# Read a CSV file
df_vital = spark.read.csv("/home/josaphat/Desktop/research/mimic-iv-ed-2.0/2.0/ed/vitalsign.csv", inferSchema=True, header=True)
df_vital = df_vital.limit(1000)


columns_to_process = ["heartrate", "resprate"]
df_vital = format_numerical(df_vital, columns_to_process)

print()