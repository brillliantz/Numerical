import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

data = pd.read_excel('E:\SYS Files\Documents\Python files\Comp.Phy\LAB_ThermalWave\data.xlsx')
print data[:10]
data.columns = ['Time', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11']
data.index = data['Time'].values
data.drop('Time', axis=1, level=None, inplace=True, errors='raise')
#data = Series(data['sales'])
data.plot(rot=10, 
          #ylim=(0,100), 
          #xlim=(0,23*12 - 1)
          )

def truncated_approximation(series, order):
  fourier = np.fft.fft(series)
  truncated_fourier = [x if i <= order or len(fourier)-i <= order else 0 for i,x in enumerate(fourier)]
  return np.real(np.fft.ifft(truncated_fourier))
data['s1.20th'] = truncated_approximation(data['s1'], order=20)
a = np.fft.fft(data['s1'])
data['s1.10th'] = truncated_approximation(data['s1'], order=10)
data.plot(x='Time', y=['s1', 's1.20th', 's1.10th'], rot=10, linewidth=2)
data.plot(x='month', y='sales', rot=10, ylim=(0,100), xlim=(0,23*12 - 1))
plt.show()