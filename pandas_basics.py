import pandas as pd
import numpy as np
from numpy.random import randn
np.random.seed(101)

labels = ['a', 'b', 'c']
my_list = [10, 20, 30]

# Series
my_series = pd.Series(index=labels, data=my_list)
print(my_series)

# DataFrames
df = pd.DataFrame(randn(5, 4), index='A B C D E'.split(), columns='W X Y Z'.split())
print("The DataFrame is:")
print(df)

# Adding a new column
df['new'] = df['W'] + df['Y']
print(df)

# Removing a column
df.drop('new', axis=1, inplace=True)
print(df)

# Selecting a row
print(df.loc['A'])

df = df[df>0]
print(df)