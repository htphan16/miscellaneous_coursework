import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

electricity = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00290/eb.arff', skiprows=12, header=None)
electricity.columns = ['forva', 'forw', 'type', 'sector', 'id']

by_type = electricity.groupby(electricity['type'])
electricity.dropna(inplace=True)
electricity = electricity.convert_objects(convert_numeric=True)
by_type = electricity.groupby('type')
forw_mean = by_type['forw'].mean()
print(forw_mean)
forva_mean = by_type['forva'].mean()
print(forva_mean)
forw_sum = by_type['forw'].sum()
print(forw_sum)
forva_sum = by_type['forva'].sum()
print(forva_sum)
pd1 = by_type['forva', 'forw'].mean()

pd2 = by_type['forva', 'forw'].sum()
pd3 = pd.concat([pd1, pd2], axis=1)
pd3.columns = ['forva_mean', 'forw_mean', 'forva_sum', 'forw_sum']
print(pd3)
pd3.plot.barh(y = ['forva_mean', 'forw_mean'], rot=0)
pd3.plot.barh(y = ['forva_sum', 'forw_sum'], rot=0)
#electricity.boxplot(vert=False, column=['forva'], by=['type'])

plt.show()
#bank = electricity[electricity[electricity.columns[2]] == 'Bank']
#print(type(bank[bank.columns[1]][1]))
#bank[bank.columns[1]])
