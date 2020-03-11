import pandas as pd
import numpy as np
import math
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as tsa
import os
import sys


df = pd.read_csv('data_matrix.csv', index_col=0, header=0)
copy_df = df.copy(deep=True)
columns = df.columns
shift = 1
target = pd.DataFrame(0, index= df.index, columns=['target'])


###############				labeling 		########################

for i in range(1,len(df.index)-1):
	if (df['Nifty_50'][df.index[i+1]] > df['Nifty_50'][df.index[i]]):
		target['target'][df.index[i]] = 1
	elif (df['Nifty_50'][df.index[i+1]] < df['Nifty_50'][df.index[i]]):
		target['target'][df.index[i]] = -1

for column in columns:
	for i in range(shift,len(df.index)):
		copy_df[column][df.index[i]] = (df[column][df.index[i]] - df[column][df.index[i-shift]]) / df[column][df.index[i-shift]]
copy_df = copy_df.drop(df.index[0])

for i in range(len(columns)):
	adf = tsa.adfuller(copy_df[columns[i]])[1]
	plt.xcorr(copy_df['Nifty_50'],copy_df[columns[i]],usevlines=False, maxlags=5, normed=False, linestyle="-", linewidth=3.0)
	print adf

copy_df = pd.concat([copy_df,target], axis=1).reindex(copy_df.index)

plt.legend(columns)
plt.axhline(color='black')
plt.ylabel("Cross Correlation")
plt.xlabel("Lag")
plt.show()
copy_df.to_csv('feature.csv')

