from pyspark.sql import DataFrame
from pyspark.sql.functions import concat, lit, collect_list, col, concat_ws, array, flatten
from pyspark.ml.feature import QuantileDiscretizer
from typing import Tuple, Dict
import pickle
import sys
sys.path.append('/home/josaphat/Desktop/research/BEHRT')
import os
from pyspark.sql import Window 
from pyspark.sql.functions import collect_list, concat, array, lit, flatten, row_number


def format_data(df: DataFrame, columns: list) -> Tuple[DataFrame, Dict[str, Dict[str, int]]]:
    """
    For each specified column in the DataFrame, check the data type. If numerical, replace numerical values
    with category based on 5 quantiles. If the data is string type, just apply the grouping and list generation.
    Then group the rows by 'subject_id' and 'stay_id', and collect the values into a list. 
    After that, add 'SEP' into each list, group by 'subject_id' again and flatten the list.

    :param df: input DataFrame
    :param columns: list of column names to process
    :return: DataFrame with updated columns and dictionary of vocabularies
    """
    # get the data types of the columns
    types = dict(df.dtypes)

    # create a dictionary to store the vocabularies
    vocab_dict = {}

    # define reserved keys
    reserved_keys = ["SEP","CLS","MASK","UNK","PAD"]

    # discretize all numerical columns and transform string columns to '_bucket' form
    for column in columns:
        if types[column] in ["int", "double", "float", "long", "short", "byte", "decimal"]:
            discretizer = QuantileDiscretizer(numBuckets=5, inputCol=column, outputCol=f"{column}_bucket")
            df = discretizer.fit(df).transform(df)

            # Convert bucket indices to string
            df = df.withColumn(f"{column}_bucket", col(f"{column}_bucket").cast("string"))
        elif types[column] == "string":
            df = df.withColumn(f"{column}_bucket", col(column))

        # build vocabulary dictionary for the processed column
        unique_vals = df.select(f"{column}_bucket").distinct().rdd.flatMap(lambda x: x).collect()
        vocab_dict[f'{column}2idx'] = {val: idx+len(reserved_keys) for idx, val in enumerate(unique_vals)}
        # prepend reserved keys
        for idx, key in enumerate(reserved_keys):
            vocab_dict[f'{column}2idx'][key] = idx



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



    # # Handling dictionary saving logic:
    # # load the existing vocab_dict.pkl if it exists, 
    # # then update or add the keys as necessary
    if os.path.exists('vocab_dict.pkl'):
        with open('vocab_dict.pkl', 'rb') as handle:
            main_vocab_dict = pickle.load(handle)
    else:
        main_vocab_dict = {}
    # Update the main vocabulary dictionary with the new vocabularies
    for column in columns:
        main_vocab_dict[f'{column}2idx'] = vocab_dict[f'{column}2idx']
        # print the number of keys in the current dictionary
        print(f"The dictionary '{column}2idx' now has {len(main_vocab_dict[f'{column}2idx'])} keys.")
    # Save the updated main vocabulary dictionary to a pickle file
    with open('vocab_dict.pkl', 'wb') as handle:
        pickle.dump(main_vocab_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pass


    return df, vocab_dict
