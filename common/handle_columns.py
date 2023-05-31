def fix_sequence_length(*args):
    def process_column(column):
        if column is None:  # Handle None values
            return [[]]
        sublists = []
        temp = []
        for item in column:
            if item == "SEP":
                sublists.append(temp)
                temp = []
            else:
                temp.append(item)
        if temp:
            sublists.append(temp)
        return sublists

    # Process all columns
    processed_columns = [process_column(column) for column in args]

    # Compute the maximum lengths
    max_length_2nd_tier = max(len(max(inner_list, key=len)) for inner_list in processed_columns)
    max_len = max(len(sublist) for sublist in processed_columns)

    # Replace [] with ['UNK']*max_length_2nd_tier
    processed_columns = [['UNK']*max_length_2nd_tier if column==[] else column for column in processed_columns]

    # Create the new 2D list with 'UNK' elements
    processed_columns_new = [[['UNK'] * max_length_2nd_tier for _ in range(max_len)] for _ in processed_columns]

    # Transfer data from old columns to new ones
    for i in range(len(processed_columns)):
        for j in range(len(processed_columns[i])):
            for k in range(len(processed_columns[i][j])):
                processed_columns_new[i][j][k] = processed_columns[i][j][k]

    final_result_columns = []
    for sublist in processed_columns_new:
        final_result = []
        for inner_list in sublist:
            final_result.extend(inner_list)
            final_result.append('SEP')
        final_result_columns.append(final_result)

    # Check if all columns have the same length
    if not all(len(column) == len(final_result_columns[0]) for column in final_result_columns):
        print('NOT SAME LENGTH')
        for column in final_result_columns:
            print(column)
            print(len(column))

    return tuple(final_result_columns)
