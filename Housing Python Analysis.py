import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv(r"C:\Users\heywo\OneDrive - Queen's University\Python\Projects\data science\Housing/train.csv")

x_variable = (df_train.columns)

#count number of columns including sales price
print(len(x_variable))

#think about the expectation and variables that impact the Salesprice

#Evaluate Depedent Variable, to see if there are abnormalies
df_train['SalePrice'].describe()

#histogram
sns.displot(df_train['SalePrice'])
plt.show()

#scatterplot and evaluate how variables correlate with SalePrice (only numerical variables)
var_1= "GrLivArea"
#use concat to create a dataframe 
data = pd.concat([df_train['SalePrice'],df_train[var_1]],axis=1)
data
data.plot.scatter(x=var_1, y='SalePrice',ylim=(0,800000))
plt.show()

#evaluate how year built impacts SalePrice
var_2 ='YearBuilt'
data = pd.concat([df_train['SalePrice'],df_train[var_2]],axis = 1)
data.plot.scatter(x=var_2, y='SalePrice',ylim =(0,800000))
plt.show()

#deal with categorical variables
var_3 = 'OverallQual'
data = pd.concat([df_train['SalePrice'],df_train[var_3]],axis=1)
data
sns.boxplot(x=df_train[var_3],y=df_train['SalePrice'],data=data)
plt.show()

#correlation matrix
correlation_matrix = df_train.corr()
f,ax= plt.subplots(figsize =(15,9))
sns.heatmap(correlation_matrix, vmax=0.8,square = True)
plt.show()

'''
#specific heatmap
k = 10
#get the strongest correlation rows
columns = correlation_matrix.nlargest(k,'SalePrice')['SalePrice'].index
print(columns)
print(np.corrcoef(df_train[columns].values))
correlation_matrix_2= np.corrcoef(df_train[columns].values)

correlation_matrix_3= np.corrcoef(df_train[columns].values).T
print(np.corrcoef(df_train[columns].values).T)

sns.set(font_scale = 1.25)
heat_map = sns.heatmap(correlation_matrix_2,cbar= True, annot= True,square= True,fmt='0.2f',annot_kws={'size':10},yticklabels=columns.values,xticklabels=columns.values)
plt.show()
'''
#scatterplot
sns.set()
#only take the highest 6 variables for correlation
columns=['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[columns],size=2)
plt.show()

#missing data
total=df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis = 1, keys =['Total','Percent'])
missing_data.head(20)

#Look at the variables that have >15% missing data, think if they are important and determine whether to drop 
df_train=df_train.drop((missing_data[missing_data['Total']>1]).index,1)

#delete one observation
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max()

#outliers
#standardizing data
#make column vectors, only make 1 dimensional array
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis])
#first 10 low range
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()[:10]]
high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print(high_range)

#you don't want anything far from 0 

#bivariate analysis
var_4 = 'GrLivArea'
data = pd.concat([df_train['SalePrice'],df_train[var_4]],axis = 1)
data.plot.scatter(x = var_4, y='SalePrice',ylim=(0,800000))
plt.show()

#drop the outliers
df_train.sort_values(by='GrLivArea',ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id']==1299].index)
df_train = df_train.drop(df_train[df_train['Id']==524].index)

#search for normality
sns.distplot(df_train['SalePrice'],fit =norm)
fig = plt.figure()
residuals = stats.probplot(df_train['SalePrice'],plot = plt)
plt.show()

#when it is not normal, we will apply log transformation
df_train['SalePrice'] = np.log(df_train['SalePrice'])

#transformed histogram
sns.distplot(df_train['SalePrice'],fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'],plot = plt)
plt.show()

#create binary variables
df_train['HasBsmt']=pd.Series(len(df_train['TotalBsmtSF']),index=df_train.index)
df_train['HasBsmt']=0
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1

#transform data, only the HasBsmt apartments
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF']=np.log(df_train['TotalBsmtSF'])

#convert categorical variables into dummy
df_train = pd.get_dummies(df_train)
df_train