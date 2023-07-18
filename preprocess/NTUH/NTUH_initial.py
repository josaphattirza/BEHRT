import numpy as np
import joblib

# data = joblib.load('/home/josaphat/Desktop/josaphat/largeXY[1216]/largeXY_BERT.joblib')

data = joblib.load('/home/josaphat/Desktop/new-josaphat/largeXY[1216]/largeXY_BERT.joblib')



joined_data = data['X'].join(data['Y'])


print("Number of rows in data: ", data['X'].shape[0])
print("Number of rows in data2: ", data['Y'].shape[0])
print("Number of rows in joined_data: ", joined_data.shape[0])


# # drop ICD, TTAS, and LABLOINC columns
# columns = [col for col in data.columns if col[:3]=='icd' or col[:4]=='ttas' or col[:8]=='LABLOINC']
# data.drop(columns=columns, inplace=True)


# # only get final TILLNOW
# IDX = np.array(data.reset_index(3).index.tolist(),dtype=object)
# lg0 = np.any(IDX[:-1]!=IDX[1:], axis=1)
# lg = np.r_[lg0, True] # last appearance
# data = data.loc[lg]
joined_data.reset_index(inplace=True)


# Assuming 'data' is your Pandas DataFrame
joined_data.head(1000).to_pickle('data_new_1000.pkl')

joined_data.to_pickle('data_new_all.pkl')





# import pandas as pd
# from sklearn.model_selection import train_test_split

# # FOR TABULAR-LM
# # Join the dataframes
# merged_data = data['X'].join(data['Y'])

# # Convert the integer columns starting with 'ttasLv2' to boolean
# for column in merged_data.columns:
#     if column.startswith('ttasLv2'):
#         merged_data[column] = merged_data[column].astype('bool')

# # Perform 80:20 train test split
# train, test = train_test_split(merged_data, test_size=0.2, random_state=42)

# # Save to pickle files
# train.to_pickle('train.pkl')
# test.to_pickle('test.pkl')