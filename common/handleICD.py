import pickle
import sys
import numpy as np
sys.path.append('/home/josaphat/Desktop/research/ED-BERT-demo')

import pandas as pd


from mimic4ed_benchmark.Benchmark_scripts.medcodes.diagnoses.icd_conversion import *
from mimic4ed_benchmark.Benchmark_scripts.medcodes.diagnoses.comorbidities import elixhauser, charlson

from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType, MapType
from pyspark.sql.functions import udf


cci_var_map = {
    'myocardial infarction': 'cci_MI',
    'congestive heart failure': 'cci_CHF',
    'peripheral vascular disease': 'cci_PVD',
    'cerebrovascular disease': 'cci_Stroke',
    'dementia': 'cci_Dementia',
    'chronic pulmonary disease': 'cci_Pulmonary',
    'rheumatic disease': 'cci_Rheumatic',
    'peptic ulcer disease': 'cci_PUD',
    'mild liver disease': 'cci_Liver1',
    'diabetes without chronic complication': 'cci_DM1',
    'diabetes with chronic complication': 'cci_DM2',
    'hemiplegia or paraplegia': 'cci_Paralysis',
    'renal disease': 'cci_Renal',
    'malignancy': 'cci_Cancer1',
    'moderate or severe liver disease': 'cci_Liver2',
    'metastatic solid tumor': 'cci_Cancer2',
    'AIDS/HIV': 'cci_HIV'
}

eci_var_map = {
'congestive heart failure' : 'eci_CHF',
'cardiac arrhythmias' : 'eci_Arrhythmia',
'valvular disease' : 'eci_Valvular',
'pulmonary circulation disorders' : 'eci_PHTN',
'peripheral vascular disorders' : 'eci_PVD',
'hypertension, complicated' : 'eci_HTN1',
'hypertension, uncomplicated' : 'eci_HTN2',
'paralysis' : 'eci_Paralysis',
'other neurological disorders' : 'eci_NeuroOther',
'chronic pulmonary disease' : 'eci_Pulmonary',
'diabetes, complicated' : 'eci_DM1',
'diabetes, uncomplicated' : 'eci_DM2',
'hypothyroidism' : 'eci_Hypothyroid',
'renal failure' : 'eci_Renal',
'liver disease' : 'eci_Liver',
'peptic ulcer disease excluding bleeding' : 'eci_PUD',
'AIDS/HIV' : 'eci_HIV',
'lymphoma' : 'eci_Lymphoma',
'metastatic cancer' : 'eci_Tumor2',
'solid tumor without metastasis' : 'eci_Tumor1',
'rheumatoid arthritis' : 'eci_Rheumatic',
'coagulopathy' : 'eci_Coagulopathy',
'obesity' : 'eci_Obesity',
'weight loss' : 'eci_WeightLoss',
'fluid and electrolyte disorders' : 'eci_FluidsLytes',
'blood loss anemia' : 'eci_BloodLoss',
'deficiency anemia' : 'eci_Anemia',
'alcohol abuse' : 'eci_Alcohol',
'drug abuse' : 'eci_Drugs',
'psychoses' : 'eci_Psychoses',
'depression' : 'eci_Depression'
}

map_dict = {'charlson': cci_var_map,
            'elixhauser': eci_var_map}
empty_map_vector = {k:{v:0 for key, v in map_i.items()} for k, map_i in map_dict.items()}


def commorbidity_set(icd, version, mapping='charlson'):
    c_list = []
    called = 0
    if mapping == 'charlson':
        for i, c in enumerate(icd):
            c_list.extend(charlson(c, version[i]))
            called+=1
    elif mapping == 'elixhauser':
        for i, c in enumerate(icd):
            c_list.extend(elixhauser(c, version[i]))
            called+=1
    else:
        raise ValueError(
            f"{mapping} is not a recognized mapping. It must be \'charlson\' or \'elixhauser\'")
    return set(c_list)


def commorbidity_dict(icd, version, mapping='charlson'):
    map_set = commorbidity_set(icd, version, mapping)
    map_vector = empty_map_vector[mapping].copy()
    for c in map_set:
        map_vector[map_dict[mapping][c]] = 1
    return map_vector


# def digit3_icd_set(icd):
#     c_list = []
#     for e in icd:
#         e = 'icd_' + e
#         c_list.append(e)
#     return set(c_list)


# def digit3_icd_dict(icd):
#     with open('./empty_code2idx.pkl','rb') as f:
#         empty_map_vector_icd = pickle.load(f)

#     map_set = digit3_icd_set(icd)
#     map_vector = empty_map_vector_icd['code2idx'].copy()
#     for c in map_set:
#         map_vector[c] = 1
#     return map_vector



# def handleICD(row, mode='3digit'):
#     if mode=='3digit':
#         return row['icd_code'][:3]
    
#     elif mode=='comorbidities':
#         cci = commorbidity_dict(row['icd_code'], 10, mapping='charlson')
#         eci = commorbidity_dict(row['icd_code'], 10, mapping='elixhauser')
#         if 'eci_1' not in row or pd.isnull(row['eci_1']):
#             return 1
#         else:
#             return row['eci_1']


def diagnosis_with_time(df_diagnoses, df_admissions):
    df_diagnoses_with_adm = pd.merge(df_diagnoses, df_admissions.loc[:, [
                                     'hadm_id', 'subject_id', 'dischtime']], on=['subject_id', 'hadm_id'], how='left')
    df_diagnoses_with_adm.loc[:, 'dischtime'] = pd.to_datetime(
        df_diagnoses_with_adm.loc[:, 'dischtime'])
    df_diagnoses_sorted = df_diagnoses_with_adm.sort_values(
        ['subject_id', 'dischtime']).reset_index()
    return df_diagnoses_sorted

# Define a function to combine values into an array
def combine_values(group):
    return pd.Series({
        'combined_icd_code': group['icd_code'].tolist(),
        'combined_icd_version': group['icd_version'].tolist()
    })


def digit3_icd_set(icd):
    return (f'icd_{e}' for e in icd)


with open('./empty_code2idx.pkl', 'rb') as f:
    empty_map_vector_icd = pickle.load(f)

def digit3_icd_dict(icd):
    map_set = digit3_icd_set(icd)
    map_vector = empty_map_vector_icd['code2idx'].copy()
    for c in map_set:
        map_vector[c] = 1
    return map_vector


from pyspark.sql import functions as F
from pyspark.sql.window import Window


def convert_9to10(code):
    if code in icd9to10dict.keys():
        return icd9to10dict[code][:3]
    else:
        return code[:3]

def commorbidity(df_main, mode='digit3'):
    # Convert all ICD 9 to 10 and get the first 3
    if mode == 'digit3':
        # Register the UDF
        map_icd_code_udf = udf(convert_9to10, StringType())

        # Apply the UDF to the "icd_code" column and save the result in a new column
        df_main = df_main.withColumn("icd_code", map_icd_code_udf(df_main["icd_code"]))

        # Group the DataFrame by 'stay_id' and 'subject_id' and apply the combine_values function
        window_spec = Window.partitionBy('stay_id', 'subject_id')
        df_main = df_main.withColumn('combined_icd_code', F.collect_list('icd_code').over(window_spec))
        df_main = df_main.withColumn('combined_icd_version', F.collect_list('icd_version').over(window_spec))

        # Drop the 'icd_code' and 'icd_version' columns
        df_main = df_main.drop('icd_code', 'icd_version')

        # Drop duplicate rows based on 'subject_id' and 'stay_id' columns
        df_main = df_main.dropDuplicates(['subject_id', 'stay_id'])

        # Merge the grouped DataFrame with the original DataFrame on 'stay_id' and 'subject_id'
        df_main = df_main.drop('combined_icd_version')

        # Alternatively, you can use the 'sortWithinPartitions' method:
        df_main = df_main.sortWithinPartitions("subject_id", "intime")

        # Drop the "combined_icd_version" column
        df_main = df_main.drop("combined_icd_version")

        # # Rename the "combined_icd_code" column to "icd_code"
        # df_main = df_main.withColumnRenamed("combined_icd_code", "icd_code")
        
        df_main = df_main.toPandas()

        print("reached here 2")

        icds = []
        for i, row in df_main.iterrows():
            icds.append({'stay_id':row['stay_id'], 
                            **digit3_icd_dict(row['combined_icd_code']),
                            })
        
        print("reached here 2,5")

        df_icds = pd.DataFrame.from_records(icds)

        print("reached here 3")

        # Sort columns based on alphabet and number, excluding the first column
        sorted_columns = [df_icds.columns[0]] + sorted(df_icds.columns[1:], key=lambda x: (x.split('_')[0], int(x.split('_')[1]) if x.split('_')[1].isdigit() else x.split('_')[1]))
        # Reorder DataFrame based on sorted columns
        df_icds = df_icds[sorted_columns]

        print("reached here 4")

        df_main = pd.merge(df_main, df_icds, on='stay_id', how='left')

    return df_main


def unique_row(M: np.ndarray): # M should be a sorted-row matrix
    lg0 = M[1:]!=M[:-1]
    lg = np.r_[True, lg0] # fist appearance
    ix = np.r_[np.nonzero(lg)[0], M.shape[0]]
    return ix, lg

def np_icd(df):
    # Convert all ICD 9 to 10 and get the first 3
    for i, row in df.iterrows():
        if row['icd_version'] == 9:
            icd10 = convert_9to10(row['icd_code'])
            df.at[i, 'icd_code'] = icd10[:3]
        elif row['icd_version'] == 10:
            df.at[i, 'icd_code'] = row['icd_code'][:3]


    df.sort_values('stay_id', inplace=True)
    unqix, unqlg = unique_row(df.stay_id.to_numpy())
    unqlg[0] = False # 0-based
    col_data = df.icd_code.copy()
    emptyvalue = 'na'
    if not isinstance(col_data.dtype, pd.CategoricalDtype):
        col_data = col_data.astype('category')
    if col_data.isna().any():
        col_data = col_data.cat.add_categories(emptyvalue).fillna(emptyvalue)
    unq = unq = np.array(['icd_' + code for code in col_data.cat.categories.to_numpy()]) # all ICD codes (unique)
    if unq.size*(unqix.size-1) <= 2**32:
        dtype = np.uint32
    else:
        dtype = np.uint64
    iv = col_data.cat.codes.to_numpy().astype(dtype)
    iv += (unq.size*np.cumsum(unqlg)).astype(dtype)
    lg = np.zeros(unq.size*(unqix.size-1), dtype=bool) # multi-hot encoding boolean array
    lg[iv] = True
    lg = lg.reshape(unqix.size-1, unq.size)
    col_stayId = df.columns.tolist().index('stay_id')
    output = pd.DataFrame(index=df.iloc[unqix[:-1],col_stayId], data=lg, columns=unq)
    return output

def np_med(df):
    df.sort_values('stay_id', inplace=True)
    unqix, unqlg = unique_row(df.stay_id.to_numpy())
    unqlg[0] = False # 0-based
    col_data = df.med.copy()
    emptyvalue = 'na'
    if not isinstance(col_data.dtype, pd.CategoricalDtype):
        col_data = col_data.astype('category')
    if col_data.isna().any():
        col_data = col_data.cat.add_categories(emptyvalue).fillna(emptyvalue)
    unq = unq = np.array(['med_' + code for code in col_data.cat.categories.to_numpy()]) # all ICD codes (unique)
    if unq.size*(unqix.size-1) <= 2**32:
        dtype = np.uint32
    else:
        dtype = np.uint64
    iv = col_data.cat.codes.to_numpy().astype(dtype)
    iv += (unq.size*np.cumsum(unqlg)).astype(dtype)
    lg = np.zeros(unq.size*(unqix.size-1), dtype=bool) # multi-hot encoding boolean array
    lg[iv] = True
    lg = lg.reshape(unqix.size-1, unq.size)
    col_stayId = df.columns.tolist().index('stay_id')
    output = pd.DataFrame(index=df.iloc[unqix[:-1],col_stayId], data=lg, columns=unq)
    return output

def icd_converter(df):
    # Convert all ICD 9 to 10 and get the first 3
    for i, row in df.iterrows():
        if row['icd_version'] == 9:
            icd10 = convert_9to10(row['icd_code'])
            df.at[i, 'icd_code'] = icd10[:3]
        elif row['icd_version'] == 10:
            df.at[i, 'icd_code'] = row['icd_code'][:3]

    return df

# def np_triage(df):
#     df.sort_values('stay_id', inplace=True)
#     unqix, unqlg = unique_row(df.stay_id.to_numpy())
#     unqlg[0] = False # 0-based
#     col_data = df.med.copy()
#     emptyvalue = 'na'
#     if not isinstance(col_data.dtype, pd.CategoricalDtype):
#         col_data = col_data.astype('category')
#     if col_data.isna().any():
#         col_data = col_data.cat.add_categories(emptyvalue).fillna(emptyvalue)
#     unq = unq = np.array(['triage_' + code for code in col_data.cat.categories.to_numpy()]) # all ICD codes (unique)
#     if unq.size*(unqix.size-1) <= 2**32:
#         dtype = np.uint32
#     else:
#         dtype = np.uint64
#     iv = col_data.cat.codes.to_numpy().astype(dtype)
#     iv += (unq.size*np.cumsum(unqlg)).astype(dtype)
#     lg = np.zeros(unq.size*(unqix.size-1), dtype=bool) # multi-hot encoding boolean array
#     lg[iv] = True
#     lg = lg.reshape(unqix.size-1, unq.size)
#     col_stayId = df.columns.tolist().index('stay_id')
#     output = pd.DataFrame(index=df.iloc[unqix[:-1],col_stayId], data=lg, columns=unq)
#     return output



# if __name__=='__main__':
#     stayId = ['visit0','visit1','visit0']
#     icd = ['A12','A12','B23']
#     df = pd.DataFrame(data={'stayId':stayId, 'ICD':icd})
#     icd_bool = commorbidity2(df)
#     print(icd_bool)



# def commorbidity(df_main, mode = 'digit3'):
#     if mode == 'commorbidities':
#         # Group the DataFrame by 'stay_id' and 'subject_id' and apply the combine_values function
#         grouped = df_main.groupby(['stay_id', 'subject_id']).apply(combine_values).reset_index(level=[0, 1])
#         # Drop the 'icd_code' and 'icd_version' columns
#         df_main = df_main.drop(columns=['icd_code', 'icd_version'])
#         # Drop duplicate rows based on 'subject_id' and 'stay_id' columns
#         df_main = df_main.drop_duplicates(subset=['subject_id', 'stay_id'])
#         # Reset the index if desired
#         df_main = df_main.reset_index(drop=True)
#         # Merge the grouped DataFrame with the original DataFrame on 'stay_id' and 'subject_id'
#         df_main = df_main.merge(grouped, on=['stay_id', 'subject_id'], how='left')


#         cci_eci=[]
    
#         for i, row in df_main.iterrows():
#             cci_eci.append({'stay_id':row['stay_id'], 
#                             **commorbidity_dict(row['combined_icd_code'], row['combined_icd_version'], mapping='charlson'), 
#                             **commorbidity_dict(row['combined_icd_code'], row['combined_icd_version'], mapping='elixhauser')})
#             pass
#         df_cci_eci = pd.DataFrame.from_records(cci_eci)
#         df_main = pd.merge(df_main, df_cci_eci, on='stay_id', how='left')
#         return df_main
    
#     elif mode == 'digit3':
#         # Convert all ICD 9 to 10 and get the first 3
#         for i, row in df_main.iterrows():
#             if row['icd_version'] == 9:
#                 icd10 = convert_9to10(row['icd_code'])
#                 df_main.at[i, 'icd_code'] = icd10[:3]
#             elif row['icd_version'] == 10:
#                 df_main.at[i, 'icd_code'] = row['icd_code'][:3]
        
#         # Group the DataFrame by 'stay_id' and 'subject_id' and apply the combine_values function
#         grouped = df_main.groupby(['stay_id', 'subject_id']).apply(combine_values).reset_index(level=[0, 1])
#         # Drop the 'icd_code' and 'icd_version' columns
#         df_main = df_main.drop(columns=['icd_code', 'icd_version'])
#         # Drop duplicate rows based on 'subject_id' and 'stay_id' columns
#         df_main = df_main.drop_duplicates(subset=['subject_id', 'stay_id'])
#         # Reset the index if desired
#         df_main = df_main.reset_index(drop=True)
#         # Merge the grouped DataFrame with the original DataFrame on 'stay_id' and 'subject_id'
#         df_main = df_main.merge(grouped, on=['stay_id', 'subject_id'], how='left')
        
#         icds = []
#         for i, row in df_main.iterrows():
#             icds.append({'stay_id':row['stay_id'], 
#                             **digit3_icd_dict(row['combined_icd_code']),
#                             })
            
#         df_icds = pd.DataFrame.from_records(icds)
#         # Sort columns based on alphabet and number, excluding the first column
#         sorted_columns = [df_icds.columns[0]] + sorted(df_icds.columns[1:], key=lambda x: (x.split('_')[0], int(x.split('_')[1]) if x.split('_')[1].isdigit() else x.split('_')[1]))
#         # Reorder DataFrame based on sorted columns
#         df_icds = df_icds[sorted_columns]

#         df_main = pd.merge(df_main, df_icds, on='stay_id', how='left')

        
#         return df_main