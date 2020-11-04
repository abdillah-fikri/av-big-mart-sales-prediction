# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent,md
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: 'Python 3.8.5 64-bit (''ds_env'': conda)'
#     metadata:
#       interpreter:
#         hash: 9147bcb9e0785203a659ab3390718fd781c9994811db246717fd6ffdcf1dd807
#     name: 'Python 3.8.5 64-bit (''ds_env'': conda)'
# ---

# %%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set(style='darkgrid', palette='muted')

# %%
raw_train = pd.read_csv('train.csv')
raw_test = pd.read_csv('test.csv')

# %% [markdown]
# | Variable | Description |
# | ------ | ----------- |
# |Item_Identifier|	Unique product ID|
# |Item_Weight|	Weight of product|
# |Item_Fat_Content|	Whether the product is low fat or not|
# |Item_Visibility|	The % of total display area of all products in a store allocated to the particular product|
# |Item_Type|	The category to which the product belongs|
# |Item_MRP|	Maximum Retail Price (list price) of the product|
# |Outlet_Identifier|	Unique store ID|
# |Outlet_Establishment_Year|	The year in which store was established|
# |Outlet_Size|	The size of the store in terms of ground area covered|
# |Outlet_Location_Type|	The type of city in which the store is located|
# |Outlet_Type|	Whether the outlet is just a grocery store or some sort of supermarket|
# |Item_Outlet_Sales|	Sales of the product in the particular store. This is the outcome variable to be predicted.|

# %%
raw_train.head()

# %%
raw_train.shape, raw_test.shape

# %%
num_col = ['Item_Weight', 'Item_Visibility', 'Item_MRP']
cat_col = ['Item_Fat_Content', 'Item_Type', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

# %%
raw_train[cat_col].nunique()

# %%
raw_train[num_col].describe()

# %%
raw_train.corr()

# %%
raw_train[cat_col].astype('str').describe(include='all')

# %%
na_col = pd.DataFrame(raw_train[num_col+cat_col].isna().sum()) / raw_train.shape[0]*100
na_col.columns = ['NA Train']
na_col['NA Test'] = raw_test[num_col+cat_col].isna().sum().values /raw_train.shape[0]*100
round(na_col, 2)

# %%
fig, ax = plt.subplots(2, 1, figsize=(12,7))
sns.histplot(raw_train.Item_Outlet_Sales, ax=ax[0])
sns.boxplot(raw_train.Item_Outlet_Sales, ax=ax[1])
ax[0].set_title('Distribusi data dari kolom target (Item_Outlet_Sales) \n', fontsize=20)
ax[0].set_xlabel('')

# %%
plt.figure(figsize=(10,7))
for i, col in enumerate(num_col):
    plt.subplot(3,1,i+1)
    sns.boxplot(raw_train[col])
    plt.xlabel('')
    plt.ylabel(col)

# %%
plt.figure(figsize=(10,7))
for i, col in enumerate(num_col):
    plt.subplot(3,1,i+1)
    sns.violinplot(raw_train[col])
    plt.xlabel('')
    plt.ylabel(col)

# %%
sns.pairplot(raw_train.drop(columns='Outlet_Establishment_Year'), hue='Outlet_Type')

# %%
plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
sns.barplot(data=raw_train, y='Item_Outlet_Sales', x='Item_Fat_Content')
plt.xlabel('Item_Fat_Content', fontsize=14)

plt.subplot(1,2,2)
sns.boxplot(data=raw_train, y='Item_Outlet_Sales', x='Item_Fat_Content')
plt.xlabel('Item_Fat_Content', fontsize=14)
plt.show()

# %%
plt.figure(figsize=(16,14))
plt.subplot(2,1,1)
sns.barplot(data=raw_train, y='Item_Outlet_Sales', x='Item_Type')
plt.xlabel('Item_Type', fontsize=14)
plt.xticks(rotation=35)

plt.subplot(2,1,2)
sns.boxplot(data=raw_train, y='Item_Outlet_Sales', x='Item_Type')
plt.xlabel('Item_Type', fontsize=14)
plt.xticks(rotation=35)
plt.show()

# %%
plt.figure(figsize=(16,10))
temp_col = ['Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
for i, col in enumerate(temp_col):
    plt.subplot(2,2,i+1)
    sns.barplot(data=raw_train, y='Item_Outlet_Sales', x=col)
    plt.xlabel(col, fontsize=14)

# %%
plt.figure(figsize=(16,10))
temp_col = ['Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
for i, col in enumerate(temp_col):
    plt.subplot(2,2,i+1)
    sns.boxplot(data=raw_train, y='Item_Outlet_Sales', x=col)
    plt.xlabel(col, fontsize=14)

# %% [markdown]
# ## Preprocessing

# %%
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve, cross_validate
from xgboost import XGBRegressor

# %%
X = raw_train[num_col+cat_col]
y = raw_train.Item_Outlet_Sales
X_test = raw_test[num_col+cat_col]

# %%
na_col = pd.DataFrame(X.isna().sum()) / X.shape[0]*100
na_col.columns = ['NA Train']
na_col['NA Test'] = X_test.isna().sum().values / X_test.shape[0]*100
round(na_col, 2)

# %% [markdown]
# Kita akan membuang variabel Outlet_Size karena missing valuenya terlalu banyak.

# %%
X.drop('Outlet_Size', axis=1, inplace=True)
X_test.drop('Outlet_Size', axis=1, inplace=True)

# %%
num_col = ['Item_Weight', 'Item_Visibility', 'Item_MRP']
cat_col = ['Item_Fat_Content', 'Item_Type', 'Outlet_Establishment_Year', 'Outlet_Location_Type', 'Outlet_Type']

numerical_transformer = SimpleImputer(strategy='median')

categorical_transformer = Pipeline(steps=[
    ('imputer2', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, num_col),
    ('cat', categorical_transformer, cat_col)
])

# %%
preprocessor.fit(X)
X_clean = preprocessor.transform(X)
X_test_clean = preprocessor.transform(X_test)

# %% tags=[]
parameters = {
                'n_estimators': [100, 200, 300, 400, 500],
                'learning_rate': [0.006, 0.01, 0.03, 0.06, 0.1]
             }

grid_search = GridSearchCV(estimator = XGBRegressor(),
                           param_grid = parameters,
                           scoring = 'neg_mean_absolute_error',
                           cv = 10)
grid_search.fit(X_clean,y)
print(grid_search.best_score_)
print(grid_search.best_params_)

# %%
submit = pd.read_csv('sample_submission.csv')
submit.columns

# %%
pred_test = grid_search.predict(X_test_clean)
output = pd.DataFrame({'Item_Identifier': raw_test['Item_Identifier'],
                       'Outlet_Identifier': raw_test['Outlet_Identifier'],
                       'Item_Outlet_Sales': pred_test})
output.to_csv('submission.csv', index=False)

# %%
