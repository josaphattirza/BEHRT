from pyspark.sql.functions import udf, col, when
from pyspark.sql.types import StringType

from mimic4ed_benchmark.Benchmark_scripts.medcodes.diagnoses._mappers.icd9to10_dict import icd9to10dict
from mimic4ed_benchmark.Benchmark_scripts.medcodes.diagnoses._mappers.icd10to9_dict import icd10to9dict
def convert_9to10(code):
    if code in icd9to10dict.keys():
        return icd9to10dict[code]
    else:
        return code

# Create a UDF (User Defined Function)
udf_convert_9to10 = udf(lambda icd_code: convert_9to10(icd_code), StringType())

def format_icd(df):
    # Use 'withColumn' to create new columns
    # You might need to drop the old column and rename the new one to the old one's name if necessary
    df = df.withColumn('icd_code', 
                       when(col('icd_version') == 9, udf_convert_9to10(col('icd_code')).substr(1, 3))
                       .otherwise(col('icd_code').substr(1, 3)))
    return df
