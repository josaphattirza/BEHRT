from pyspark.sql import DataFrame
from pyspark.sql.functions import concat, lit, collect_list, col, concat_ws, array, flatten
from pyspark.ml.feature import QuantileDiscretizer

def format_data(df: DataFrame, columns: list) -> DataFrame:
    """
    For each specified column in the DataFrame, check the data type. If numerical, replace numerical values
    with category based on 5 quantiles. If the data is string type, just apply the grouping and list generation.
    Then group the rows by 'subject_id' and 'stay_id', and collect the values into a list. 
    After that, add 'SEP' into each list, group by 'subject_id' again and flatten the list.

    :param df: input DataFrame
    :param columns: list of column names to process
    :return: DataFrame with updated columns
    """
    # get the data types of the columns
    types = dict(df.dtypes)

    # discretize all numerical columns and transform string columns to '_bucket' form
    for column in columns:
        if types[column] in ["int", "double", "float", "long", "short", "byte", "decimal"]:
            discretizer = QuantileDiscretizer(numBuckets=5, inputCol=column, outputCol=f"{column}_bucket")
            df = discretizer.fit(df).transform(df)

            # Convert bucket indices to string
            df = df.withColumn(f"{column}_bucket", col(f"{column}_bucket").cast("string"))
        elif types[column] == "string":
            df = df.withColumn(f"{column}_bucket", col(column))

    # Group by 'subject_id' and 'stay_id', and collect values into a list
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
