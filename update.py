#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import csv
import json
import requests
from datetime import datetime


# In[2]:


date = pd.to_datetime("today").strftime('_%m_%d')
print('Latest update time is:',date)


# In[3]:


states = pd.read_csv("https://api.covid19india.org/csv/latest/states.csv")

# change column names to lowercase
states.columns= states.columns.str.lower()

# convert date column
states['date'] = pd.to_datetime(states['date'], format= '%Y-%m-%d')


# In[4]:


do_not_include = ['India','State Unassigned']


# In[5]:


states


# In[6]:


# NOTE: 
# The data set appears to add entries for the most recent date then fill in information as it comes in, 
# meaning a state will appear to have zero new cases for the most recent day until it gets updated.

# filter out most recent date to avoid potentially incomplete information
states = states[~(states['date'] == states['date'].max())]


# In[7]:


states


# In[8]:


# pivot data with states as columns
pivot_cases = pd.pivot_table(states, index = "date", columns = "state", values= "confirmed")

# drop non-state columns
pivot_cases = pivot_cases.drop(columns=do_not_include)

# replacing 0 total cases with nan
#pivot_cases.replace(0, np.nan, inplace=True)


# In[9]:


pivot_cases


# In[10]:


# new dataframe to store "daily new cases"
pivot_newcases = pivot_cases.copy()

# calculate "daily new cases"
for column in pivot_newcases.columns[0:]:
    DailyNewCases = column
    pivot_newcases[DailyNewCases] = pivot_newcases[column].diff()


# In[11]:


# fill NaN in pivot_newcases (first row) with values from pivot_cases
pivot_newcases.fillna(pivot_cases, inplace=True)


# In[12]:


pivot_newcases


# In[13]:


# replace negative daily values by setting 0 as the lowest value
pivot_newcases = pivot_newcases.clip(lower=0)


# In[14]:


# new dataframe to store "avg new cases"
pivot_avgnewcases = pivot_newcases.copy()

# calculate 7-day averages of new cases
for column in pivot_avgnewcases.columns[0:]:
    DaySeven = column
    pivot_avgnewcases[DaySeven] = pivot_avgnewcases[column].rolling(window=7, center=False).mean()


# In[15]:


# fill NaN in pivot_avgnewcases (first 6 rows) with values from pivot_newcases
pivot_recentnew = pivot_avgnewcases.fillna(pivot_newcases)


# In[16]:


pivot_recentnew


# In[17]:


# new dataframe to store "avg new cases" with centered average
pivot_avgnewcases_center = pivot_newcases.copy()

# calculate 7-day averages of new cases with centered average
for column in pivot_avgnewcases_center.columns[0:]:
    DaySeven = column
    pivot_avgnewcases_center[DaySeven] = pivot_avgnewcases_center[column].rolling(window=7, min_periods=4, center=True).mean()


# In[18]:


pivot_avgnewcases_center


# In[19]:


# reset indexes of "pivoted" data
pivot_cases = pivot_cases.reset_index()
pivot_newcases = pivot_newcases.reset_index()
pivot_recentnew = pivot_recentnew.reset_index()
pivot_avgnewcases_center = pivot_avgnewcases_center.reset_index()


# In[20]:


# convert "pivot" of total cases to "long form"
state_cases = pd.melt(pivot_cases, id_vars=['date'], var_name='state', value_name='cases')


# In[21]:


state_cases


# In[22]:


# convert "pivot" of daily new cases to "long form"
state_newcases = pd.melt(pivot_newcases, id_vars=['date'], var_name='state', value_name='new_cases')


# In[23]:


state_newcases


# In[24]:


# convert "pivot" of recent new cases to "long form" (7-day avg w first 6 days from "new cases")
state_recentnew = pd.melt(pivot_recentnew, id_vars=['date'], var_name='state', value_name='recent_new')


# In[25]:


state_recentnew


# In[26]:


# convert "pivot" of centered average new cases to "long form"
state_avgnewcases_center = pd.melt(pivot_avgnewcases_center, id_vars=['date'], var_name='state', value_name='avg_cases')


# In[27]:


state_avgnewcases_center


# In[28]:


# merge the 4 "long form" dataframes based on index
state_merge = pd.concat([state_cases, state_newcases, state_avgnewcases_center, state_recentnew], axis=1)


# In[29]:


state_merge


# In[30]:


# remove duplicate columns
state_merge = state_merge.loc[:,~state_merge.columns.duplicated()]


# In[31]:


# dataframe with only the most recent date for each state
# https://stackoverflow.com/questions/23767883/pandas-create-new-dataframe-choosing-max-value-from-multiple-observations
state_latest = state_merge.loc[state_merge.groupby('state').date.idxmax().values]


# In[32]:


state_latest


# In[33]:


# dataframe with peak average cases for each state
peak_avg_cases = state_merge.groupby('state')['recent_new'].agg(['max']).reset_index()
peak_avg_cases = peak_avg_cases.rename(columns = {'max':'peak_recent_new'})


# In[34]:


# merging total cases onto the merged dataframe
state_color_test = state_latest.merge(peak_avg_cases, on='state', how='left')


# In[35]:


state_color_test


# In[36]:


#choosing colors
n_0 = 20
f_0 = 0.5
f_1 = 0.2

# https://stackoverflow.com/questions/49586471/add-new-column-to-python-pandas-dataframe-based-on-multiple-conditions/49586787
def conditions(state_color_test):
    if state_color_test['recent_new'] <= n_0*f_0 or state_color_test['recent_new'] <= n_0 and state_color_test['recent_new'] <= f_0*state_color_test['peak_recent_new']:
        return 'green'
    elif state_color_test['recent_new'] <= 1.5*n_0 and state_color_test['recent_new'] <= f_0*state_color_test['peak_recent_new'] or state_color_test['recent_new'] <= state_color_test['peak_recent_new']*f_1:
        return 'orange'
    else:
        return 'red'

state_color_test['color'] = state_color_test.apply(conditions, axis=1)


# In[37]:


state_color_test


# In[38]:


# dataframe with just state, total cases, and color
state_total_color = state_color_test[['state','cases','color']]

# rename cases to total_cases for the purpose of merging
state_total_color = state_total_color.rename(columns = {'cases':'total_cases'})


# In[39]:


# merging total cases onto the merged dataframe
state_final = state_merge.merge(state_total_color, on='state', how='left')


# In[40]:


state_final = state_final[['state','date','cases','new_cases','avg_cases','total_cases','recent_new','color']]


# In[48]:


state_final[state_final['state'] == 'Sikkim']


# In[42]:


# drop rows where cumulative cases is NaN (dates before reported cases)
state_final = state_final.dropna(subset=['cases']) 


# In[43]:


state_final


# In[44]:


## Remove the 'cases' column to match format of Era's state result file 
state_final = state_final[['state','date','new_cases','avg_cases','total_cases','recent_new','color']]

state_final.to_csv('result.csv', index=False)


# In[45]:


# dataframe with just state and color
state_color = state_color_test[['state','color']]

# creates csv similar to USStateColors.csv
state_color.to_csv('stateColors.csv', index=False)

