import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.fftpack import dct

# Reading data from the file specified
df = pd.read_csv(r'Table1.txt', delim_whitespace=True, header=None)

# Transforming values to numeric
df = df.apply(pd.to_numeric, errors='coerce')
# Displaying the data from dataset
print("Here is the data of Table1.txt - ")
print(df)
# Displaying the original dimensions
print("Let's look at original data dimensionality of Table1.txt - ", df.shape)


# Let's use a function to reduce dimensionality using principal component analysis
def pca_dim_reduce(df, variance_threshold=0.98):
    # Showcasing at least 98% of total variance in original data
    pca_r = PCA(n_components=variance_threshold)
    df_pca_r = pca_r.fit_transform(df)
    return df_pca_r


df_pca_r = pca_dim_reduce(df)

# Displaying reduced dimensionality after we have applied principal component analysis
print("Let's look at dimensionality after we applied PCA on Table1.txt:", df_pca_r.shape[1])


# Let's use a function to reduce dimensionality using Discrete cosine transform

def dct_dime_reduce(df):
    df_dct_r = dct(df.values, axis=0, norm='forward')
    # Setting the threshold
    thrshld = np.percentile(np.abs(df_dct_r), 98)
    # Coefficient below threshold will be set to 0
    df_dct_r[np.abs(df_dct_r) < thrshld] = 0
    # Find out how many non-zero coefficient exist
    nz_cfft = np.count_nonzero(df_dct_r, axis=0)
    final_dim = np.count_nonzero(nz_cfft)
    return final_dim


final_dim = dct_dime_reduce(df)
print("Let's look at dimensionality after we applied DCT on Table1.txt:", final_dim)

# Following same process for Table2.txt dataset
df2 = pd.read_csv(r'Table2.txt', delim_whitespace=True, header=None)
df2 = df2.apply(pd.to_numeric, errors='coerce')
# Displaying the data from dataset
print("Here is the data of Table2.txt - ")
print(df2)
# Displaying the original dimensions
print("Let's look at original data dimensionality of Table2.txt - ", df2.shape)

# PCA on Table2.txt
df_pca_r2 = pca_dim_reduce(df2)
print("Let's look at dimensionality after we applied PCA on Table2.txt:", df_pca_r2.shape[1])

# DCT on Table2.txt
final_dim2 = dct_dime_reduce(df2)
print("Let's look at dimensionality after we applied DCT on Table2.txt:", final_dim2)

# Following same process for Table3.txt
df3 = pd.read_csv(r'Table3.txt', delim_whitespace=True, header=None)
df3 = df3.apply(pd.to_numeric, errors='coerce')
# Displaying the data from dataset
print("Here is the data of Table3.txt - ")
print(df3)
# Displaying the original dimensions
print("Let's look at original data dimensionality of Table3.txt - ", df2.shape)

# PCA on Table3.txt
df_pca_r3 = pca_dim_reduce(df3)
print("Let's look at dimensionality after we applied PCA on Table3.txt:", df_pca_r3.shape[1])

# DCT on Table3.txt
final_dim3 = dct_dime_reduce(df3)
print("Let's look at dimensionality after we applied DCT on Table3.txt:", final_dim3)
