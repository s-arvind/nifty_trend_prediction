import pandas as pd
import numpy as np
from collections import defaultdict
import statsmodels.tsa.stattools as tsa
import os
import sys

pd.set_option('display.width', 1000)
path = "data/"
file_names = []

for filename in os.listdir(path):
	file_names.append(filename)

df = pd.read_csv(path+'Nifty_50.csv', header=0, index_col=0)
df = df[::-1]
df = pd.to_numeric(df['Price'].replace(r',','',regex=True))
columns = ['Nifty_50']

for name in file_names:
	if name != 'Nifty_50.csv':
		file = pd.read_csv(path+name, header=0, index_col=0)
		file = file[::-1]
		file = pd.to_numeric(file['Price'].replace(r',','',regex=True))
		df = pd.concat([df,file], axis=1).reindex(df.index)							
		columns.append(name.split('.')[0])

df.columns = columns
df = df.interpolate()
df.to_csv("data_matrix.csv")



