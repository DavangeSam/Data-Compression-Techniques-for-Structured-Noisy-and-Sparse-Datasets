import pandas as pd
import umap
import matplotlib.pyplot as plt

# Reading dataset from Table1.txt

df = pd.read_csv(r'Table1.txt', delim_whitespace=True, header=None)

# Transforming values to numeric
df = df.apply(pd.to_numeric, errors='coerce')
# Displaying the data from dataset
print("Here is the data of Table1.txt - ")
print(df)
# Displaying the original dimensions
print("Let's look at original data dimensionality of Table1.txt - ", df.shape)
# Applying Uniform Manifold Approximation and Projection
umap_dim_r = umap.UMAP(n_components=2, random_state=42).fit_transform(df)
# Displaying dimensions after applying umap
print("Let's look at the dimensionality after applying UMAP:", umap_dim_r.shape[1])

# Following same process for Table2.txt
df2 = pd.read_csv(r'Table2.txt', delim_whitespace=True, header=None)
df2 = df2.apply(pd.to_numeric, errors='coerce')
print("Here is the data of Table2.txt - ")
print(df2)
print("Let's look at original data dimensionality of Table2.txt - ", df2.shape)
umap_dim_r2 = umap.UMAP(n_components=2, random_state=42).fit_transform(df2)
print("Let's look at the dimensionality after applying UMAP:", umap_dim_r2.shape[1])

# Following same process for Table3.txt
df3 = pd.read_csv(r'Table3.txt', delim_whitespace=True, header=None)
df3 = df3.apply(pd.to_numeric, errors='coerce')
print("Here is the data of Table3.txt - ")
print(df3)
print("Let's look at original data dimensionality of Table3.txt - ", df3.shape)
umap_dim_r3 = umap.UMAP(n_components=2, random_state=42).fit_transform(df3)
print("Let's look at the dimensionality after applying UMAP:", umap_dim_r3.shape[1])

