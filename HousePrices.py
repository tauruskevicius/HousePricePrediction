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
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

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

#Outliers - data indicated that there are some
fix, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.show()

#remove outliers
train = train.drop(train[(train['SalePrice'] < 200000) & (train['GrLivArea'] > 4000)].index)

#check data again
fix, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.show()

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

#Outliers - data indicated that there are some
fix, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.show()

#Missing data
total = train.isnull().sum().sort_values(ascending = False)
percent = (train.isnull().sum()/train.isnull().count() * 100).sort_values(ascending = False)
na_table = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percentage'])
na_table = na_table[na_table['Total'] != 0] #remove where no empty values

#visualization of missing data
f, ax = plt.subplots()
plt.xticks(rotation = '90')
plt.gcf().subplots_adjust(bottom = 0.2) #make room for x labels 
plot = sns.barplot(x = na_table.index, y = na_table['Percentage'])
#annotate bars
for p in plot.patches:
    plot.annotate(format(p.get_height(), '.1f'),
                  (p.get_x() + p.get_width() / 2., p.get_height()),
                  ha = 'center', va = 'center',
                  size = 15,
                  xytext = (0, 9),
                  textcoords = 'offset points')
plt.xlabel('Features', fontsize = 15)
plt.ylabel('Missing value percentage', fontsize = 15)
plt.title('Percent missing data by feature', fontsize = 15)
plt.show()

#correlation map to see how features are correlated with SalePrice
corr = train.corr()
plt.subplots(figsize = (12,9))
plt.gcf().subplots_adjust(bottom = 0.2) #make room for x labels 
sns.heatmap(corr, vmax=0.9, square = True)

#Dealing with missing values. Trying to prevent data leakage by not concatanating
#the training and testing data sets. 

train['PoolQC'] = train['PoolQC'].fillna('None') #none means there is no pool
test['PoolQC'] = test['PoolQC'].fillna('None')

train['MiscFeature'] = train['MiscFeature'].fillna('None') #None means no misc feature
test['MiscFeature'] = test['MiscFeature'].fillna('None')

train['Alley'] = train['Alley'].fillna('None') #None means no alley access
test['Alley'] = test['Alley'].fillna('None')

train['Fence'] = train['Fence'].fillna('None') #None means no fence
test['Fence'] = test['Fence'].fillna('None')

train['FireplaceQu'] = train['FireplaceQu'].fillna('None') #None means no fireplace
test['FireplaceQu'] = test['FireplaceQu'].fillna('None')

#Street length of the house should be similar to other houses in the same neighborhood
#Grouped data by neighborhood and filled with median from each neighborhood
train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].transform(
                        lambda x: x.fillna(x.median()))
test['LotFrontage'] = test.groupby('Neighborhood')['LotFrontage'].transform(
                        lambda x: x.fillna(x.median())) #usde test data to fill the values


#loop to fill certain columns with 'None'
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    train[col] = train[col].fillna('None')
    test[col] = test[col].fillna('None')

#loop to fill columns with 0 --- no garage = no yearsblt, no area, no cars inside
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)
    
#basement missing values due to no basement
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
            'BsmtHalfBath') :
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    train[col] = train[col].fillna('None')
    test[col] = test[col].fillna('None')

#MassVnr type and area 
train['MasVnrType'] = train['MasVnrType'].fillna('None')
test['MasVnrType'] = test['MasVnrType'].fillna('None')
train['MasVnrArea'] = train['MasVnrArea'].fillna(0)
test['MasVnrArea'] = test['MasVnrArea'].fillna(0)

#last missing value from train
train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])  

#check for missing values
train.isnull().sum() #0
test.isnull().sum() #still a few left

#filter columns with missing values
test_na = test.columns[test.isnull().any()]
test_na
    
#fill the missing values using train data
test['MSZoning'] = test['MSZoning'].fillna(train['MSZoning'].mode(0)[0])

#only 2 missing values, mostly all utilities
test['Utilities'] = test['Utilities'].fillna(train['Utilities'].mode()[0])

#only 1 missing value
test['Exterior1st'] = test['Exterior1st'].fillna(train['Exterior1st'].mode()[0])
test['Exterior2nd'] = test['Exterior2nd'].fillna(train['Exterior2nd'].mode()[0])
 
#1 missing value, fill with mode   
test['KitchenQual'] = test['KitchenQual'].fillna(train['KitchenQual'].mode()[0])
    
#2 missing values, fill with typical
test['Functional'] = test['Functional'].fillna('Typ')

#only a few missing values, fill with mode
test['SaleType'] = test['SaleType'].fillna(train['SaleType'].mode()[0]) 
 
#check if any empty values left
train.isnull().any()    
test.isnull().any()    
#no more empty values

#Change some numerical values that do not make sense numerically
#Years, OverallCond, MSSubClass
train['MSSubClass'] = train['MSSubClass'].apply(str)
test['MSSubClass'] = test['MSSubClass'].apply(str)

train['OverallCond'] = train['OverallCond'].apply(str)
test['OverallCond'] = test['OverallCond'].apply(str)

train['YrSold'] = train['YrSold'].astype(str)
train['MoSold'] = train['MoSold'].astype(str)
test['YrSold'] = test['YrSold'].astype(str)
test['MoSold'] = test['MoSold'].astype(str)

#label encoder categorical variables
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

#encountering an issue with label encoder when fitting with only the training set
#the test set contains unseen values and thus cannot be encoded
#will have to jon two sets and separate after encoding

#joining data
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
df = pd.concat((train, test)).reset_index(drop = True)
df.drop(['SalePrice'], axis = 1, inplace = True)

#transforming 
for col in cols:
    le = LabelEncoder()
    le.fit(list(df[col].values))
    df[col] = le.transform(list(df[col].values))

#adding total squre footage of the house as a new feature
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

#splitting data back
train = df[:ntrain]
test = df[ntrain:]

#checking the skewness of numerical features
numeric_features = df.dtypes[df.dtypes != 'object'].index

skewed_features = df[numeric_features].apply(lambda x: skew(x.dropna())).sort_values(ascending = False)

skewness = pd.DataFrame({'Skew': skewed_features})

#box cox transformation for skewed data

skewness = skewness[abs(skewness.Skew) > 0.75] #focus on highly skewed data
print('There are {} skewed numerical features to Box Cox transform'.format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_feat = skewness.index
lambd = 0.15
for feat in skewed_feat:
    train[feat] = boxcox1p(train[feat], lambd)
    test[feat] = boxcox1p(test[feat], lambd)
    
#qq plot to see the skewness of a random transformed variable
fig = plt.figure()
res = stats.probplot(train['LotArea'], plot = plt)
plt.show() #more normally distributed

#get dummy categorical features
train = pd.get_dummies(train)
test = pd.get_dummies(test)

#MODELING


















