#!/usr/bin/env python
# coding: utf-8

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Loading data from an Excel file
df = pd.read_excel('your_file.xlsx')

# Displaying the first 10 rows of the dataframe
df.head(10)

# Sorting the 'Total' column for ECDF calculation
x = np.sort(df['Total'])

# Finding the length of the sorted data
len(x)

# Calculating the cumulative distribution frequency
y = np.arange(1, len(x)+1) / len(x)

# Plotting the Empirical Cumulative Distribution Function (ECDF)
plt.plot(x, y, marker='.')
plt.xlabel('Percentage of Sale Price for Units Sold')
plt.ylabel('Cumulative Distribution Frequency')
plt.title('ECDF of Sale Price')

# Calculating the correlation matrix for the dataframe
df_cor = df.corr()
df_cor

# Extracting 'Total' as the independent variable (x) and 'UnitCost' as the dependent variable (y) for linear regression
x = df['Total']
y = df['UnitCost']

# Fitting a linear regression model
ml_model = LinearRegression().fit(x, y)

# Checking the model's score (R^2 value)
ml_model.score()

# Creating new data for a simple linear regression example
x = np.array([12,3,4,5,6,7]).reshape((-1,1))
y = np.array([4,7,9,10,11,20])

# Displaying the independent variable data
x

# Initializing a new linear regression model
reg = LinearRegression()

# Fitting the model with the new data
reg = LinearRegression().fit(x,y)

# Calculating and printing the R^2 score of the model
r_score = reg.score(x,y)
print(r_score)

# Extracting and printing the intercept of the linear regression model
intercept = reg.intercept_
print(intercept)

# Extracting and displaying the coefficient (slope) of the model
co_eff = reg.coef_
co_eff

# Making predictions using the model and displaying them
prediction = reg.predict(x)
print(prediction)

# Importing additional libraries for further analysis
import pandas as pd
import scipy.stats as stats

# Loading a dataset from a CSV file
df = pd.read_csv('your_file.csv')
df.head()

# Dropping rows with missing values
df = df.dropna()

# Checking for any remaining missing values
df.isnull().sum()

# Extracting weights of athletes from different teams
us_team = df[df['Team']== 'US'].Weight
netherlands_team = df[df['Team']== 'Netherlands'].Weight
denmark_team = df[df['Team']== 'Denmark'].Weight

# Performing one-way ANOVA to compare weights among the teams
anova = stats.f_oneway(us_team, netherlands_team, denmark_team)
print(anova)

# Creating arrays for weights of athletes from specific countries
France_athletes = df[df.Team == "France"].Weight
US_athletes = df[df.Team == "United States"].Weight
China_athletes = df[df.Team == "China"].Weight

# Performing one-way ANOVA for these countries
anova = stats.f_oneway(France_athletes, US_athletes, China_athletes)
print(anova)

# Importing necessary library for another approach to ANOVA
from scipy.stats import f_oneway

# Creating arrays for weights of athletes from specific countries
performance1 = df[df.Team == "France"].Weight
performance2 = df[df.Team == "United States"].Weight
performance3 = df[df.Team == "China"].Weight

# Conducting the one-way ANOVA
f_oneway(performance1, performance2, performance3)

# Importing statsmodels for another statistical analysis approach
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Defining a formula for ANOVA with 'Weight' as the dependent variable and 'Sex' and 'Team' as independent variables
formula = 'Weight ~ Sex + Team'
model = ols(formula, data=df).fit()

# Generating and displaying the ANOVA table
aov_table = sm.stats.anova_lm(model, typ=2)
print(aov_table)
