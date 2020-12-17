# House Prices Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math

import warnings
warnings.filterwarnings('ignore')
#Importing dataset
os.chdir('C:\\Users\\Owner\\Desktop\\DataScience\\Machine Learning Practice\\MyGitHub\\Regression')

df = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', None)
#EDA
print(df.shape)
print(df.info())

df.isnull().sum()

#EDA
import matplotlib.style as style
from spicy import stats
style.use('fivethirtyeight')
sns.distplot(df['SalePrice'])
plt.title('Distribution of House Sale Price')
plt.show()

