# Project_2_Machine_Learning_AMV

# This project uses Maricopa County Tax Assesor's data and Supervised Machine Learning techniques to develope an automated valuation model based on the cash value figure reported by the tax assessor and the physical features and locations of homes. # 


## First, any relevant libraries are imported here. ##

```python
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error
```

<a href="https://imgur.com/sG22Tv5"><img src="https://i.imgur.com/sG22Tv5.jpg" title="source: imgur.com" /></a>

## Inorder to normalize and tee up the data for the machine learning process we used the get_dummies operation for both construction year and zip code. ##

```python
rf_df = ma_df[['SitusZip','LandSize', 'LivableSqFootage', 'ConstructionYear', 'Pool', 'FullCashValue']].copy()
rf_df = pd.get_dummies(rf_df, columns=["SitusZip", 'ConstructionYear'])
rf_df = rf_df[rf_df.FullCashValue != 0]
```

## We need to establish a target for the model to attempt to predict. For this particular valuation model we've decided to use the "Full Cash Value" figures provided by the county tax assessor's office.  ##
 - First we drop the column from our dataset then we establish it as our target value.
```python

X = rf_df.copy()
X.drop("FullCashValue", axis=1, inplace=True)

y = rf_df["FullCashValue"].values.reshape(-1, 1)
y[:5]
```

## Here we are training the model based off the two split data sets that were formed above. ##
 
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)
scaler = StandardScaler()
X_scaler = scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_fit = rf_model.fit(X_train_scaled, y_train.ravel())
predictions = rf_fit.predict(X_test_scaled)
```

## Lets see what are the most important attributes are in determing the full cash value. ##

```python
[(0.4647436496635477, 'LivableSqFootage'),
 (0.2464154409298598, 'LandSize'),
 (0.033440283386282375, 'ConstructionYear_2002'),
 (0.03322693374371636, 'SitusZip_85032.0'),
 (0.01982897673119421, 'SitusZip_85006.0'),
 (0.0148564424240925, 'SitusZip_85016.0'),
 (0.009557882758952273, 'SitusZip_85040.0'),
 (0.00904575126139606, 'ConstructionYear_2019'),
 (0.008255666873345231, 'SitusZip_85008.0'),
 (0.007005156176119154, 'SitusZip_85035.0')]
```
### Based on the results above Livable Square Footage and Lot Size were the best predictors. ###

## We want a high R^2 and as low as possible a score for our RMSE score. ##

```python
R^2 score:                 0.77
RMSE score:       502,073,729.14
```
### This is an acceptbale starting point for our model and shows some promise for further testing and tweaking.###



## In order to continue to refine our model we'll need to check and control for outliers. ##

```python
sns.boxplot(x=ma_df['FullCashValue'])
```

<a href="https://imgur.com/kF9Ykhh"><img src="https://i.imgur.com/kF9Ykhh.jpg" title="source: imgur.com" /></a>

```python
ma_df = ma_df[ma_df.FullCashValue != 0]
sns.boxplot(x=ma_df['FullCashValue'])
``` 
<a href="https://imgur.com/Y32zm4I"><img src="https://i.imgur.com/Y32zm4I.jpg" title="source: imgur.com" /></a>


```python
cols = ['FullCashValue']

Q1 = ma_df[cols].quantile(0.25)
Q3 = ma_df[cols].quantile(0.75)
IQR = Q3 - Q1

ma_df = ma_df[~((ma_df[cols] < (Q1 - 1.5 * IQR)) |(ma_df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

sns.boxplot(x=ma_df['FullCashValue'])
```

<a href="https://imgur.com/dpRlcC5"><img src="https://i.imgur.com/dpRlcC5.jpg" title="source: imgur.com" /></a>
 - After removing some of the outliers in the process above we now have a decent sized normalized dataset.

## Lets check to see if there is a trend for the parameters we're interested in.##

```python
sns.pairplot(ma_df[["LivableSqFootage", "ConstructionYear", "LandSize", 'FullCashValue' ]], diag_kind="kde")
```

<a href="https://imgur.com/Ln92tSX"><img src="https://i.imgur.com/Ln92tSX.jpg" title="source: imgur.com" /></a>


## Splitting our X & Y, Normalizing the Data, and splitting it up into training and testing versions. ##

```python
X = ma_df[['LivableSqFootage', 'ConstructionYear', 'LandSize', 'Pool']]
y = ma_df['FullCashValue']

X_normalized = preprocessing.normalize(X, norm='l2')

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, random_state=42)
```

## Running our model. ##

```python
regressor = LinearRegression(normalize=True)

regressor.fit(X_train, y_train)

predictions = regressor.predict(X_test)
print(f'R^2 score: {r2_score(y_true=y_test, y_pred=predictions):20,.2f}')
print(f'RMSE score: {mean_squared_error(y_true=y_test, y_pred=predictions, squared=True):20,.2f}'
```
### What were the scores? ###

 ```python
R^2 score:                 0.63
RMSE score:       749,345,484.10
```

### So as we can see, the linear regression model is not as good at predicting price. Anyone who has worked with SKLearn already has a pretty good idea that Random Forest will be hard to beat in most situations. ##
