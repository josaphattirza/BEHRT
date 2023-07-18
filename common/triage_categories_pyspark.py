from pyspark.sql.functions import col, when, isnan

def categorize_temp(df, col_name):
    return df.withColumn(col_name, 
                         when((isnan(df[col_name])) | (df[col_name] == 0), 'UNK')
                         .when(df[col_name] < 95, 'Temp-Hypothermia')
                         .when(df[col_name] < 95.5, 'Temp-Low')
                         .when(df[col_name] < 99.5, 'Temp-Normal')
                         .when(df[col_name] < 100.9, 'Temp-Low grade fever')
                         .when(df[col_name] < 103, 'Temp-Fever')
                         .otherwise('Temp-High fever'))

# And similarly for the other functions...
def categorize_hr(df, col_name):
    return df.withColumn(col_name, 
                         when((isnan(df[col_name])) | (df[col_name] == 0), 'UNK')
                         .when(df[col_name] < 60, 'Hr-Bradycardia')
                         .when(df[col_name] < 100, 'Hr-Normal')
                         .when(df[col_name] < 120, 'Hr-Mild tachycardia')
                         .when(df[col_name] < 150, 'Hr-Moderate tachycardia')
                         .otherwise('Hr-Severe tachycardia'))

def categorize_rr(df, col_name):
    return df.withColumn(col_name, 
                         when((isnan(df[col_name])) | (df[col_name] == 0), 'UNK')
                         .when(df[col_name] < 12, 'Rr-Bradypnea')
                         .when(df[col_name] < 20, 'Rr-Normal')
                         .when(df[col_name] < 24, 'Rr-Mild tachypnea')
                         .when(df[col_name] < 30, 'Rr-Moderate tachypnea')
                         .otherwise('Rr-Severe tachypnea'))

def categorize_o2sat(df, col_name):
    return df.withColumn(col_name,
                         when((isnan(df[col_name])) | (df[col_name] == 0), 'UNK')
                         .when(df[col_name] <= 80, 'O2sat-Severe hypoxemia')
                         .when(df[col_name] <= 90, 'O2sat-Moderate hypoxemia')
                         .when(df[col_name] <= 94, 'O2sat-Mild hypoxemia')
                         .when(df[col_name] <= 97, 'O2sat-Normal')
                         .otherwise('O2sat-Normal to high'))

def categorize_sbp(df, col_name):
    return df.withColumn(col_name,
                         when((isnan(df[col_name])) | (df[col_name] == 0), 'UNK')
                         .when(df[col_name] >= 180, 'Sbp-Severe hypertension')
                         .when(df[col_name] >= 160, 'Sbp-Stage 2 hypertension')
                         .when(df[col_name] >= 140, 'Sbp-Stage 1 hypertension')
                         .when(df[col_name] >= 120, 'Sbp-Prehypertension')
                         .otherwise('Sbp-Normal blood pressure'))

def categorize_dbp(df, col_name):
    return df.withColumn(col_name,
                         when((isnan(df[col_name])) | (df[col_name] == 0), 'UNK')
                         .when(df[col_name] >= 100, 'Dbp-Stage 2 hypertension')
                         .when(df[col_name] >= 90, 'Dbp-Stage 1 hypertension')
                         .when(df[col_name] >= 80, 'Dbp-Prehypertension')
                         .when(df[col_name] >= 60, 'Dbp-Normal')
                         .otherwise('Dbp-Low'))

def categorize_pain(df, col_name):
    return df.withColumn(col_name,
                         when((isnan(df[col_name])) | (df[col_name] == 0), 'UNK')
                         .when(df[col_name] == 0, 'Pain-No pain')
                         .when(df[col_name] <= 3, 'Pain-Mild pain')
                         .when(df[col_name] <= 6, 'Pain-Moderate pain')
                         .when(df[col_name] <= 9, 'Pain-Severe pain')
                         .when((df[col_name] >= 10) & (df[col_name] <= 13), 'Pain-Very severe pain')
                         .otherwise('UNK'))

def categorize_acuity(df, col_name):
    return df.withColumn(col_name,
                         when((isnan(df[col_name])) | (df[col_name] == 0), 'UNK')
                         .when(df[col_name] == 1, 'Acuity-1')
                         .when(df[col_name] == 2, 'Acuity-2')
                         .when(df[col_name] == 3, 'Acuity-3')
                         .when(df[col_name] == 4, 'Acuity-4')
                         .when(df[col_name] == 5, 'Acuity-5')
                         .otherwise('UNK'))
