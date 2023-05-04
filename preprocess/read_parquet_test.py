import pandas as pd

data = pd.read_parquet('./behrt_fixed_disposition_med_month_based_test')
# data = pd.read_parquet('./behrt_format_mimic4ed_month_based_test')


print(data)