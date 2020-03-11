from sklearn import svm
import pandas as pd
import numpy as np
import math
from pandas.tools.plotting import autocorrelation_plot
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as tsa
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


path = 'data/'
file_names = []

# for filename in os.listdir(path):
# 	file_names.append(filename)
# df = pd.read_csv(path+'nifty.csv', index_col=0, header=0)
# df = pd.to_numeric(df['Close'].replace(r',','',regex=True))
# columns = ['nifty']

# for name in file_names:
# 	if name != 'nifty.csv':
# 		file = pd.read_csv(path+name, header=0, index_col=0)
# 		file = pd.to_numeric(file['Close'].replace(r',','',regex=True))
# 		df = pd.concat([df,file], axis=1).reindex(df.index)							
# 		columns.append(name.split('.')[0])


# df.columns = columns
# df = df.interpolate()
# df.to_csv("data_matrix.csv")

df = pd.read_csv('data_matrix.csv', index_col=0, header=0)
columns = df.columns

shift = 1
label = pd.DataFrame(1, index= df.index, columns=['label'])

for i in range(len(df.index)-1):
	if (df['nifty'][df.index[i+1]] >= df['nifty'][df.index[i]]):
		label['label'][df.index[i]] = 1
	else:
		label['label'][df.index[i]] = 0

# copy_df = df.copy(deep=True)

# for column in columns:
# 	for i in range(shift,len(df.index)):
# 		copy_df[column][df.index[i]] = (df[column][df.index[i]] - df[column][df.index[i-shift]]) / df[column][df.index[i-shift]]

# copy_df = copy_df.drop(df.index[0])
legend = []
# for i in range(len(columns)):
# 	adf = tsa.adfuller(copy_df[columns[i]])[1]
# 	plt.xcorr(copy_df['nifty'],copy_df[columns[i]],usevlines=False, maxlags=5, normed=False, linestyle="-", linewidth=2.0)
# 	legend.append(columns[i])
# 	print adf

# copy_df = pd.concat([copy_df,label], axis=1).reindex(copy_df.index)

# plt.legend(legend)
# plt.axhline(color='black')
# plt.axvline(color='black')
# plt.ylabel("Cross Correlation")
# plt.xlabel("Lag")
# plt.show()
# copy_df.to_csv('feature.csv')
# feature = pd.DataFrame(data=df['nifty'].tolist(),index=df.index,columns=['nifty'])
# feature = df['nifty']
# feature = feature.drop(feature.index[0])
# for column in columns:
# 	if column != 'nifty':
# 		print column
# 		temp = df[column].drop(df.index[len(df.index)-1])
# 		feature[column] = temp.tolist()

copy_df = df.copy(deep=True)
# print copy_df
for column in columns:
	for i in range(1,len(df.index)):
		copy_df[column][df.index[i]] = (df[column][df.index[i]] - df[column][df.index[i-1]]) / df[column][df.index[i-1]]
copy_df = copy_df.drop(copy_df.index[0])

# for i in range(len(copy_df.index)):
# 	norm = np.linalg.norm(copy_df.ix[copy_df.index[i]])
# 	copy_df.ix[copy_df.index[i]] /= norm

for i in range(len(columns)):
	adf = tsa.adfuller(copy_df[columns[i]])[1]
	plt.xcorr(copy_df['nifty'],copy_df[columns[i]],usevlines=False, maxlags=5, normed=False, linestyle="-", linewidth=2.0)
	legend.append(columns[i])
	print adf
copy_df = pd.concat([copy_df,label], axis=1).reindex(copy_df.index)
plt.legend(legend)
plt.axhline(color='black')
plt.axvline(color='black')
plt.ylabel("Cross Correlation")
plt.xlabel("Lag")
plt.show()
copy_df.to_csv('feature.csv')