import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as stt
import scipy.signal as signal
import seaborn as sns

data = pd.read_excel('E:\SYS Files\Documents\Python files\Comp.Phy\LAB_ThermalWave\data.xlsx')
#print data[:10]
data.columns = ['Time', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11']
data.index = data['Time'].values
data.drop('Time', axis=1, level=None, inplace=True, errors='raise')
data_cut = data.iloc[800:, :]
data.plot(rot=10, 
          #ylim=(0,100), 
          #xlim=(0,23*12 - 1)
          )
cut = 2100
s1 = data_cut.loc[cut:, 's1']
ss1 = s1 - s1.mean()
############################acf pacf analysis stationary
#ss1_acf = stt.acf(ss1, nlags=len(ss1))
#ss1_pacf = stt.pacf(ss1, nlags=300)
#ss1_acovf = stt.acovf(ss1)
#ss1_std_acovf = np.sqrt(ss1_acovf)
#plt.figure(dpi=300)
#plt.plot(ss1)
#plt.figure(dpi=300)
#plt.plot(ss1_acf, label='acf')
#plt.plot(ss1_pacf, label='pacf')
#plt.legend(loc='best')

############### ÉáÆú 
#ss1_perio = signal.periodogram(ss1)
#plt.semilogy(ss1_perio[0], ss1_perio[1])
#plt.ylim([1e2, 1e8])
#plt.xlabel('frequency [Hz]')
#plt.ylabel('PSD [V**2/Hz]')
#plt.show()

############################## FFT Spectral analysis
'''
When the input a is a time-domain signal and A = fft(a), 
np.abs(A) is its amplitude spectrum and np.abs(A)**2 is its power spectrum. 
The phase spectrum is obtained by np.angle(A).
'''
signal = ss1.values #take out array
fourier = np.fft.rfft(signal) # rfft only need to compute half because of symmetry
fff = (np.abs(fourier)) ** 2 #square of abs. so this is energy spectrum
n = signal.size
sample_rate = 1.  # sample rate of SENSOR
freq = np.fft.rfftfreq(n, d=1./sample_rate) # ÆµÂÊarray
plt.figure(figsize=(20,10),dpi=300)
plt.semilogy(freq, fff) # LOG SCALE
plt.ylim([1e3, 1e9])

################## Wave Filtering
order = 10
truncated_fourier = np.array([x if i <= order and i >= 4 else 0 for i,x in enumerate(fourier)])
signal_filted = np.fft.irfft(truncated_fourier, len(signal))
plt.figure(dpi=300)
plt.semilogy(freq, fff, lw=1) # LOG SCALE
plt.semilogy(freq, (np.abs(truncated_fourier)) ** 2, lw=2)
plt.ylim([1e3, 1e10])

plt.figure(dpi=300)
plt.plot(signal, lw=1)
plt.plot(signal_filted, lw=2)
#plt.plot(np.zeros_like(signal), lw=2)
plt.show()
