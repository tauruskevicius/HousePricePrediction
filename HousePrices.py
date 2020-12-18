# House Prices Regression
#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import os
import math
import missingno as msno #missing values
from scipy import stats
from scipy.stats import skew, norm

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

#spyder configurations
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', None)

#Importing dataset
os.chdir('C:\\Users\\Owner\\Desktop\\DataScience\\Machine Learning Practice\\MyGitHub\\Regression')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Initial look at data
train.head()
train.info()
print(train.shape)

#Data Processing

#Look at target variable SalePrice
sns.distplot(train['SalePrice'], fit = norm)

#Get fitted parameters (mean, st. dev.)
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
           loc = 'best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get quantile-quantile plot to check for normality
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot = plt)
plt.show() #distr. is skewed right
#need to normalize data for better linear model performance

#Log transformation of target variable
train['SalePrice'] = np.log1p(train['SalePrice'])

#Look at target variable SalePrice now
sns.distplot(train['SalePrice'], fit = norm)

#Get fitted parameters (mean, st. dev.)
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
           loc = 'best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get quantile-quantile plot to check for normality
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot = plt)
plt.show() #data is more normally distributed

#removing Id column 
train_Id = train['Id']
test_Id = test['Id']

train.drop('Id', axis = 1, inplace = True)
test.drop('Id', axis = 1, inplace = True)

#Missing data
total = train.isnull().sum().sort_values(ascending = False)
percent = (train.isnull().sum()/train.isnull().count() * 100).sort_values(ascending = False)
na_table = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percentage'])
na_table = na_table[na_table['Total'] != 0] #remove where no empty values







