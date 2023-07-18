from pyspark.sql import DataFrame
from pyspark.sql.functions import concat, lit, collect_list, col, concat_ws, array, flatten
from pyspark.ml.feature import QuantileDiscretizer
from typing import Tuple, Dict
import pickle
import sys
sys.path.append('/home/josaphat/Desktop/research/ED-BERT-demo')
import os
from pyspark.sql import Window 
from pyspark.sql.functions import collect_list, concat, array, lit, flatten, row_number, col, concat, split, explode



def format_data(df: DataFrame, columns: list, csv_columns: list) -> Tuple[DataFrame, Dict[str, Dict[str, int]]]:
    types = dict(df.dtypes)
    vocab_dict = {}
    reserved_keys = ["SEP", "CLS", "MASK", "UNK", "PAD"]

    for column in columns:
        if types[column] in ["int", "double", "float", "long", "short", "byte", "decimal"]:
            discretizer = QuantileDiscretizer(numBuckets=5, inputCol=column, outputCol=f"{column}_bucket")
            df = discretizer.fit(df).transform(df)
            df = df.withColumn(f"{column}_bucket", col(f"{column}_bucket").cast("string"))
        elif types[column] == "string":
            df = df.withColumn(f"{column}_bucket", col(column))
        
        unique_vals = df.select(f"{column}_bucket").distinct().rdd.flatMap(lambda x: x).collect()

        vocab = {key: idx for idx, key in enumerate(reserved_keys)}
        unique_vals = [val for val in unique_vals if val != 'UNK']
        for idx, val in enumerate(unique_vals):
            vocab[val] = idx + len(reserved_keys)

        vocab_dict[f'{column}2idx'] = vocab

    for csv_column in csv_columns:
        df = df.withColumn(csv_column, split(col(csv_column), ","))
        unique_vals = df.select(explode(col(csv_column))).distinct().rdd.flatMap(lambda x: x).collect()

        vocab = {key: idx for idx, key in enumerate(reserved_keys)}
        unique_vals = [val for val in unique_vals if val != 'UNK']
        for idx, val in enumerate(unique_vals):
            vocab[val] = idx + len(reserved_keys)

        vocab_dict[f'{csv_column}2idx'] = vocab
        df = df.withColumn(f"{csv_column}_bucket", col(csv_column))

    exprs = [collect_list(f"{column}_bucket").alias(f"{column}_bucket") for column in columns + csv_columns]
    df = df.groupBy("subject_id", "stay_id").agg(*exprs)

    # Process the normal columns and csv_columns separately
    for column in columns:
        df = df.withColumn(f"{column}_bucket", concat_ws(',', col(f"{column}_bucket"), lit("SEP")))
    for column in csv_columns:
        df = df.withColumn(f"{column}_bucket", flatten(col(f"{column}_bucket")))
        df = df.withColumn(f"{column}_bucket", concat_ws(',', col(f"{column}_bucket"), lit("SEP")))

    df = df.groupBy("subject_id")
    exprs = [collect_list(f"{column}_bucket").alias(f"{column}_bucket") for column in columns + csv_columns]
    df = df.agg(*exprs)

    for column in columns + csv_columns:
        df = df.withColumn(f"{column}_bucket", concat_ws(',', col(f"{column}_bucket")))
        df = df.withColumn(f"{column}_bucket", split(col(f"{column}_bucket"), ","))


    if os.path.exists('vocab_dict.pkl'):
        with open('vocab_dict.pkl', 'rb') as handle:
            main_vocab_dict = pickle.load(handle)
    else:
        main_vocab_dict = {}

    for column in columns + csv_columns:
        main_vocab_dict[f'{column}2idx'] = vocab_dict[f'{column}2idx']
        print(f"The dictionary '{column}2idx' now has {len(main_vocab_dict[f'{column}2idx'])} keys.")

    with open('vocab_dict_NTUH.pkl', 'wb') as handle:
        pickle.dump(main_vocab_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return df, vocab_dict
