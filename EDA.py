#!/usr/bin/env python
# coding: utf-8

# In[11]:


get_ipython().run_line_magic('pylab', '--no-import-all inline')
import pandas as pd                   
import numpy as np                 
import matplotlib.pyplot as plt    
import matplotlib.ticker as ticker  
import datetime as dt              
import matplotlib.dates as mdates     
from sklearn import preprocessing;
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn import linear_model;
import seaborn as sns
pd.options.display.float_format = '{:.2f}'.format


#importing dataset
data = pd.read_csv(r'data.csv')
data.head()

#----------------------------------------------------------------------------

#Exploratory Data Analysis

# Explore nulls...
data.isnull().sum()

# Explore where the nulls are...
data.loc[data.isnull().sum(axis = 1).astype(bool)].nunique()

# Get dates...
dates = data.loc[data.isnull().sum(axis = 1).astype(bool)]['date'].unique()

# Drop...
data.drop(np.arange(len(data))[data['date'].isin(dates)], inplace = True)

# Re-check for nulls...
data.isnull().sum()

# Preview Data...
data.head()

#Exploring 500 listed companies :
# Verify count of each company using np.unique...
# Using a nested np.unique like this returns the unique count of days in
# the first array and the count of those unique days in the second array
np.unique(np.unique(data['Name'], return_counts = True)[1], return_counts = True)
# Verify number of companies...
data['Name'].nunique()
#Of course the bottom companies listed will have to fight to remain in the top 500 S&P...let's drop these companies!
loc = data['Name'].isin(np.unique(data['Name'])[np.unique(data['Name'], return_counts = True)[1] < 250])
data.drop(loc.index[loc], inplace = True)
# Re-verify count of each company...
np.unique(np.unique(data['Name'], return_counts = True)[1], return_counts = True)

#Let's compare how each one did to day 1 (cumulative compound growth).
# Obtain all names to corresponding close...
name, close = data[data['date'] == data['date'].min()][['Name', 'close']].T.values

# Map to dictionary...
base = {n : c for n, c in zip(name, close)}

# Use base to add Growth to data...
base2 = 2
data['Growth'] = data['close'] / base2 - 1
# Summary statistics...
data['Growth'].describe()

# change to an integer of your choice
# large n will cause the later visualisation to slow down
n = 10

# last day of the year for all companies only showing Date, Name and Growth sorted by Growth...
obj = data.loc[251::250][['date', 'Name', 'Growth']].sort_values('Growth')

print('Largest fall in share price:')
display(obj.head(n))
print('Largest gain in share price:')
display(obj.tail(n))

#Worst and Best 10 performing Companies
lo_names = obj.head(n)['Name']
hi_names = obj.tail(n)['Name']
names    = [lo_names, hi_names]
pre      = ['Worst', 'Best']
title    = ' {} Performing Companies in the S&P 500'.format(n)

for i in range(2):
    # subset of data...
    sub = data[data['Name'].isin(names[i])].groupby(['Name', 'date'])['Growth'].sum().reset_index()
    
    # create figure...
    plt.figure(figsize(20, 10))
    
    sns.pointplot(x = 'date', y = 'Growth', hue = 'Name', data = sub, scale = 0.3)
    
    # remove the 250 x ticks...
    plt.xticks([])
    
    # add title and x / y labels 
    plt.title(pre[i] + title  , size = 25)
    plt.xlabel('Time (1 Year)', size = 20)
    plt.ylabel('Growth'       , size = 20)
    
    # always add a legend and make sure it is readable
    plt.legend(markerscale = 2, prop = {'size' : 15})
    
    plt.show()

#correlation
cor = data.pivot('date', 'Name', 'Growth').corr()
sns.heatmap(cor)
plt.show()
#This is very hard to read so let's focus on strong positive correlation then strong negative correlation.
def get_correlation(cor, upper = 0.99, lower = -0.99, blank = False):
    corr = {}
    for i in cor:
        obj = cor[i][cor[i].apply(abs).argsort()[::-1]]
        obj = obj[(obj > upper) | (obj < lower)]
        obj = obj[obj.index != i]
        if len(obj) == 0:
            if blank: corr[i] = None
        else:
            corr[i] = {}
            for j in obj.index:
                corr[i][j] = cor[i].loc[j]
    return corr

corr = get_correlation(cor, lower = -1)
corr

sns.heatmap(cor.loc[list(corr), list((corr))], annot = True, fmt = '.2f')
plt.xlabel('')
plt.ylabel('')
plt.show()

# And try for negative correlation...
corr = get_correlation(cor, upper = 1, lower = -0.95)
corr

sns.heatmap(cor.loc[list(corr), list((corr))], annot = True, fmt = '.2f')
plt.xlabel('')
plt.ylabel('')
plt.show()


# In[ ]:




