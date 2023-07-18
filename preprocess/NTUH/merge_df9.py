import joblib
import numpy as np

data = joblib.load('/home/josaphat/Desktop/josaphat/largeXY[1216]/largeXY_All[3171].joblib')

df9 = joblib.load('/home/josaphat/Desktop/josaphat/NTUH-iMD[1216]/09_急診診斷.joblib')

X = data['X']

IDX = np.array(X.reset_index(3).index.tolist(),dtype=object)
lg0 = np.any(IDX[:-1]!=IDX[1:], axis=1)
lg = np.r_[lg0, True] # last appearance
X = X.loc[lg]
X.reset_index(level=3, inplace=True)

df9 = df9.loc[df9.IODIAGNOSISCODE=='O','ICD10']

columns = [col for col in X.columns if col[:3]=='icd']
X.drop(columns=columns, inplace=True)
lg = X.index.isin(df9.index)
X.loc[lg,'ICD10'] = df9.loc[X.index[lg],'ICD10']
# newX = X.merge(df9, on=['PERSONID2','ACCOUNTIDSE2','ACCOUNTSEQNO'], how='inner')


X.reset_index(inplace=True)