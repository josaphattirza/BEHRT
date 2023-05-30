from pyspark.sql import DataFrame
from pyspark.sql.functions import concat, lit, collect_list, col, concat_ws, array, flatten
from pyspark.ml.feature import QuantileDiscretizer

def format_numerical(df: DataFrame, columns: list) -> DataFrame:
    """
    For each specified column in the DataFrame, replace numerical values
    with category based on 5 quantiles. Then group the rows by 'subject_id' and 'stay_id',
    and collect the bucketed categories into a list.

    :param df: input DataFrame
    :param columns: list of column names to process
    :return: DataFrame with updated columns
    """
    # discretize all columns
    for column in columns:
        discretizer = QuantileDiscretizer(numBuckets=5, inputCol=column, outputCol=f"{column}_bucket")
        df = discretizer.fit(df).transform(df)

        # Convert bucket indices to string
        df = df.withColumn(f"{column}_bucket", col(f"{column}_bucket").cast("string"))

    # Group by 'subject_id' and 'stay_id', and collect bucketed categories into a list
    exprs = [collect_list(f"{column}_bucket").alias(f"{column}_bucket") for column in columns]
    df = df.groupBy("subject_id", "stay_id").agg(*exprs)


    for column in columns:
        df = df.withColumn(f"{column}_bucket", concat(col(f"{column}_bucket"), array(lit("SEP"))))

    # Group by 'subject_id' again and aggregate the bucketed categories into a list
    df = df.groupBy("subject_id")
    exprs = [collect_list(f"{column}_bucket").alias(f"{column}_bucket") for column in columns]
    df = df.agg(*exprs)
    
    # Flatten each list
    for column in columns:
        df = df.withColumn(f"{column}_bucket", flatten(col(f"{column}_bucket")))


    return df
