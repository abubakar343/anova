#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[2]:


df = pd.read_excel('data1.xlsx')


# In[3]:


df.head(10)


# In[4]:


x = np.sort(df['Total'])


# In[5]:


len(x)


# In[6]:


y = np.arange(1, len(x)+1) / len(x)


# In[7]:


graph = plt.plot(x, y, marker='.')
graph = plt.xlabel('Percentage of Sale Price for Units Sold')
graph = plt.ylabel('Cumulative Distribution Frequency')
graph = plt.title('ECDF of Sale Price')


# In[8]:


df_cor = df.corr()
df_cor


# In[9]:


x = df['Total']
y = df['UnitCost']

ml_model =LinearRegression().fit(x,y)
# In[14]:


ml_model.score()


# In[18]:


x = np.array([12,3,4,5,6,7]).reshape((-1,1))
y = np.array([4,7,9,10,11,20])


# In[19]:


x


# In[20]:


reg = LinearRegression()


# In[39]:


reg  = LinearRegression().fit(x,y)


# In[29]:


r_score = reg.score(x,y)
print(r_score)


# In[32]:


intercept = reg.intercept_
print(intercept)


# In[34]:


co_eff = reg.coef_
co_eff


# In[36]:


prediction = reg.predict(x)
print(prediction)


# In[19]:


import pandas as pd
import scipy.stats as stats


# In[4]:


df = pd.read_csv('athlete_events.csv')
df.head()


# In[35]:


df = df.dropna()


# In[36]:


df.isnull().sum()


# In[25]:


us_team = df[df['Team']== 'US'].Weight
netherlands_team = df[df['Team']== 'Netherlands'].Weight
denmark_team = df[df['Team']== 'Denmark'].Weight


# In[38]:


anova = stats.f_oneway(us_team, netherlands_team, denmark_team)
print(anova)


# In[39]:


# Create arrays
France_athletes = df[df.Team == "France"].Weight
US_athletes = df[df.Team == "United States"].Weight
China_athletes = df[df.Team == "China"].Weight

# Perform one-way ANOVA
anova = stats.f_oneway(France_athletes, US_athletes, China_athletes)
print(anova)


# In[40]:



# Importing library
from scipy.stats import f_oneway
 
# Performance when each of the engine
# oil is applied
performance1 = df[df.Team == "France"].Weight
performance2 = df[df.Team == "United States"].Weight
performance3 = df[df.Team == "China"].Weight

 
# Conduct the one-way ANOVA
f_oneway(performance1, performance2, performance3)


# In[59]:


import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[66]:


formula = 'Weight ~ Sex + Team'
model = ols(formula, data=df).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print(aov_table)


# In[ ]:




