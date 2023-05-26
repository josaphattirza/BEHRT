import numpy as np
import pandas as pd

def unique_row(M: np.ndarray): # M should be a sorted-row matrix
    lg0 = M[1:]!=M[:-1]
    lg = np.r_[True, lg0] # fist appearance
    ix = np.r_[np.nonzero(lg)[0], M.shape[0]]
    return ix, lg

def icd2boolean(df):
    df.sort_values('stayId', inplace=True)
    unqix, unqlg = unique_row(df.stayId.to_numpy())
    unqlg[0] = False # 0-based
    col_data = df.ICD.copy()
    emptyvalue = 'na'
    if not isinstance(col_data.dtype, pd.CategoricalDtype):
        col_data = col_data.astype('category')
    if col_data.isna().any():
        col_data = col_data.cat.add_categories(emptyvalue).fillna(emptyvalue)
    unq = col_data.cat.categories.to_numpy() # all ICD codes (unique)
    if unq.size*(unqix.size-1) <= 2**32:
        dtype = np.uint32
    else:
        dtype = np.uint64
    iv = col_data.cat.codes.to_numpy().astype(dtype)
    iv += (unq.size*np.cumsum(unqlg)).astype(dtype)
    lg = np.zeros(unq.size*(unqix.size-1), dtype=bool) # multi-hot encoding boolean array
    lg[iv] = True
    lg = lg.reshape(unqix.size-1, unq.size)
    col_stayId = df.columns.tolist().index('stayId')
    output = pd.DataFrame(index=df.iloc[unqix[:-1],col_stayId], data=lg, columns=unq)
    return output


if __name__=='__main__':
    stayId = ['visit0','visit1','visit0']
    icd = ['A12','A12','B23']
    df = pd.DataFrame(data={'stayId':stayId, 'ICD':icd})
    icd_bool = icd2boolean(df)
    print(icd_bool)
