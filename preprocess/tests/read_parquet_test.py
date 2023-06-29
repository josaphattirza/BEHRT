import pandas as pd
from pyspark.sql import SparkSession


# spark = SparkSession.builder.getOrCreate()

data = pd.read_parquet('./behrt_triage_revisit_disposition_med_month_based_train')

# filtered_data = data[data['subject_id'] == 10000032]

# print(filtered_data)


# distribution = data['label'].value_counts()

# print(distribution)

print(data.count())