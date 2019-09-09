# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 01:04:06 2019

@author: Deep
"""


#https://www.kaggle.com/gdaley/hkracing

import matplotlib as mpl
import pandas as pd
from datetime import datetime
from pandas_datareader import data
import matplotlib.pyplot as plt
import seaborn as sns

print (datetime.today().now())

#function for getting stock data
def get_data (symbol, start_date, end_date):
    dat = data.DataReader(symbol, "yahoo", start_date, end_date)
    return dat
    
#list of stock code
code = ["^HSI","0005.hk","0019.hk","0293.hk","1299.hk"]

s_date = datetime(2000,1,1)
e_date = datetime.today()

df_dict = {}
for c in code[0:11]:
    try:
        df = get_data (c, s_date, e_date)
        df_dict[c] = df
        print(c)
    except:
        print("")

#data processing        
dHSI = df_dict["^HSI"]      
d0005 = df_dict["0005.hk"]
d0019 = df_dict["0019.hk"]
d0293 = df_dict["0293.hk"]
d1299 = df_dict["1299.hk"]

dff = pd.concat([dHSI,d0005,d0019,d0293,d1299], axis = 1)
dff.columns
dff.columns = ['HSI_High', 'HSI_Low', 'HSI_Open', 'HSI_Close', 'HSI_Volume', 'HSI_Adj Close', 
               '0005_High', '0005_Low', '0005_Open', '0005_Close', '0005_Volume', '0005_Adj Close', 
               '0019_High', '0019_Low', '0019_Open', '0019_Close', '0019_Volume', '0019_Adj Close', 
               '0293_High', '0293_Low', '0293_Open', '0293_Close', '0293_Volume', '0293_Adj Close', 
               '1299_High', '1299_Low', '1299_Open', '1299_Close', '1299_Volume', '1299_Adj Close']


dff1 = dff[['HSI_Adj Close', '0005_Adj Close', '0019_Adj Close', 
               '0293_Adj Close', '1299_Adj Close']]
dff_ma05 = dff1.rolling(window=50).mean()
dff_ma05.columns = [c+" moving average (50)" for c in dff_ma05.columns]
dff_ma10 = dff1.rolling(window=100).mean()
dff_ma10.columns = [c+" moving average (100)" for c in dff_ma10.columns]
dff_ma20 = dff1.rolling(window=200).mean()
dff_ma20.columns = [c+" moving average (200)" for c in dff_ma20.columns]
dff_ma50 = dff1.rolling(window=500).mean()
dff_ma50.columns = [c+" moving average (500)" for c in dff_ma50.columns]
dff_ma05.columns

dff = pd.concat([dff,dff_ma05,dff_ma10,dff_ma20,dff_ma50], axis = 1)


#Stock Price Movement of 0019.HK and 0923.HK
#https://matplotlib.org/users/colors.html
fig, ax1 = plt.subplots(figsize=(20,10))
ax2 = ax1.twinx()
#ax1.plot(d0019['Close'])
ax1.plot(dff['0019_Adj Close'], color='crimson',)
ax1.plot(dff['0019_Adj Close moving average (500)'],'--', color='chocolate', alpha = 0.5)
ax2.plot(dff['0293_Adj Close'], color='darkblue')
ax2.plot(dff['0293_Adj Close moving average (500)'],'--', color='navy', alpha = 0.5)

ax1.set_xlabel('Time')
ax1.set_ylabel('$ of 0019.HK', color='crimson')
ax2.set_ylabel('$ of 0293.HK', color='darkblue')
plt.title('Stock Price Trend for 0019.HK and 0293.HK')
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax1.legend(['Close Price of 0019.HK','Moving Average (500) of 0019.HK'],
            bbox_to_anchor=(0.2, 0.95), frameon=False)
ax2.legend(['Close Price of 00293.HK','Moving Average (500) of 0019.HK'],
            bbox_to_anchor=(0.2,0.88), frameon=False)


plt.show()




#ax = dff.plot.scatter('0293_Adj Close', '0019_Adj Close')
#ax = dff.plot.scatter('0293_Adj Close', '0019_Adj Close', c='HSI_Adj Close',s = 10, colormap='viridis', alpha = 0.5)
#ax.set_aspect('equal')







#sns.jointplot(dff['0293_Adj Close'], dff['0019_Adj Close'], kind='hex');

#sns.jointplot(dff['0293_Adj Close'], dff['0019_Adj Close'], kind='kde', space=0);

#plot= sns.kdeplot(dff['0293_Adj Close']);
#plot= sns.kdeplot(dff['0019_Adj Close']);

import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


x = dff['0293_Adj Close']
y = dff['0019_Adj Close']
z = dff['HSI_Adj Close']
fig = plt.figure(figsize=(15,10))
gs = gridspec.GridSpec(3, 3)

ax0 = plt.subplot(gs[0, 1:])
plt.hist(x, bins=100, normed=True)
ax1 = plt.subplot(gs[1:, 0])
plt.hist(y, bins=100, orientation='horizontal', normed=True)
ax2 = plt.subplot(gs[1:, 1:])
cm = plt.cm.get_cmap('viridis')
plt.scatter(x, y, s=10 ,c=z, cmap=cm, alpha = 0.5)

fig.tight_layout()

cbaxes = inset_axes(ax2, width="3%", height="80%", loc=4) 
cbar = plt.colorbar(cax=cbaxes, ticks=[0,10000,20000,30000])
#cbar.ax.set_yticklabels(['8000','0','35000'])
plt.clim(1)
ax0.set_title('Distribution of 0293.HK')
ax1.set_title('Distribution of 0019.HK')
#plt.title('Scatter Plot of 0293.HK and 0019.HK')
ax2.set_title('Scatter Plot of 0293.HK and 0019.HK')
ax1.invert_xaxis()

plt.show()



############################################################################################################3
############################################################################################################3
############################################################################################################3
############################################################################################################3
############################################################################################################3
############################################################################################################3















#Data Visualization
#Histogram and Correlation
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
fig = plt.figure(figsize=(15,10))
gs = gridspec.GridSpec(3, 3)


#cmap = matplotlib.cm.get_cmap('viridis')
#normalize = matplotlib.colors.Normalize(vmin=min(dff['HSI_Adj Close']), vmax=max(dff['HSI_Adj Close']))
#colors = [cmap(normalize(value)) for value in dff['HSI_Adj Close']]

plot_0_1 = plt.subplot(gs[0, 1:])
plt.hist(dff['0293_Adj Close'], bins=100, normed=True)
plot_1_0 = plt.subplot(gs[1:, 0])
plt.hist(dff['0019_Adj Close'], bins=100, orientation='horizontal', normed=True)
plot_1_1 = plt.subplot(gs[1:, 1:])
cm = plt.cm.get_cmap('RdYlBu_r')
plot_1_1.scatter(dff['0293_Adj Close'], dff['0019_Adj Close'],
                 c=dff['HSI_Adj Close']/max(dff['HSI_Adj Close']),s = 10,cmap = cm, alpha = 0.5)

cbaxes = inset_axes(plot_1_1, width="3%", height="80%", loc=4) 
plt.colorbar(cax=cbaxes, ticks=[0.,2])


#plot_1_1.legend(cax)
#plt.colorbar(plot_1_1)
#Set colour map and scale
plot_0_1.set_title('Distribution of 0293.HK')
plot_1_0.set_title('Distribution of 0019.HK')
#plt.title('Scatter Plot of 0293.HK and 0019.HK')
plot_1_1.set_title('Scatter Plot of 0293.HK and 0019.HK')
plot_1_0.invert_xaxis()

plt.show()
