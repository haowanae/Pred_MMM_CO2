"""
Created on Wed Nov  8 15:35:28 2023

@author: Hao WAN
"""
## Import the required libraries
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate as val,KFold
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import joblibconda
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

## Load the dataset
data = pd.read_excel( r"C:\Users\17718\Desktop\新建文件夹\表\data4.xlsx", sheet_name="A+B+C+D")
data[data.isnull()]
data[data.notnull()]
dataset = data
x = dataset.iloc[:, 0:11].values
x = x.astype(np.float64)
y = dataset.iloc[:, 11].values
y = y.astype(np.float64)

##normalization
x_MinMax = preprocessing.MinMaxScaler()
y = np.array(y).reshape(len(y), 1)
X = x_MinMax.fit_transform(x)
##Divide the data set
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=10)

## Training Stacking Algorithm model
model_1 = RandomForestRegressor(random_state=1)
model_2 = XGBRegressor(random_state=1)
meta_model = XGBRegressor(random_state=1)

train_predictions_1 = np.zeros((len(X), ))
train_predictions_2 = np.zeros((len(X), ))

n_folds = 5
kf = KFold(n_splits=n_folds, random_state=1, shuffle=True)

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model_1.fit(X_train, y_train.ravel())
    model_2.fit(X_train, y_train.ravel())

    train_predictions_1[test_index] = model_1.predict(X_test)
    train_predictions_2[test_index] = model_2.predict(X_test)

feature_matrix = np.column_stack((train_predictions_1, train_predictions_2))
meta_model.fit(feature_matrix, y)

test_predictions_1 = model_1.predict(X_test)
test_predictions_2 = model_2.predict(X_test)
test_feature_matrix = np.column_stack((test_predictions_1, test_predictions_2))
test_predictions = meta_model.predict(test_feature_matrix)

train_predictions_1 = model_1.predict(X_train)
train_predictions_2 = model_2.predict(X_train)
train_feature_matrix = np.column_stack((train_predictions_1, train_predictions_2))
train_predictions = meta_model.predict(train_feature_matrix)

##Save the normalized converter
joblib.dump(x_MinMax, "model/scaler.pkl")

## save model
os.makedirs('model', exist_ok=True)
joblib.dump(model_1, "model/model_11.pt")
joblib.dump(model_2, "model/model_12.pt")
joblib.dump(meta_model, "model/meta_model1.pt")
#print(os.getcwd())

# Evaluate performance of stacked model on training set
print("Performance on training set:")
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_train, train_predictions)))
print("MAE:", metrics.mean_absolute_error(y_train, train_predictions))
R2_train = metrics.r2_score(y_train, train_predictions)
print("R^2:", R2_train)

# Evaluate performance of stacked model on test set
print("Performance on test set:")
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, test_predictions)))
print("MAE:", metrics.mean_absolute_error(y_test, test_predictions))
R2_test = metrics.r2_score(y_test, test_predictions)
print("R^2:", R2_test)


# Plot results
xx = range(0, len(y_test[::50]))
plt.figure(figsize=(8, 6))
plt.scatter(xx, y_test[::50], color="red", label="Sample Point", linewidth=3)
plt.plot(xx, test_predictions[::50], color="orange",label="Fitting Line", linewidth=2)
plt.legend()
plt.show()
plt.figure()
plt.scatter(y_test, test_predictions)
plt.show()

# Plot feature importances
importances = model_1.feature_importances_
model_importances = pd.Series(importances)
fig, ax = plt.subplots()
model_importances.plot.bar(ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

## k-fold cross-validation
cv = KFold(n_splits=5 ,shuffle=True,random_state=1402)
scores= val(meta_model,x,y,cv=cv,return_train_score=True
                   ,scoring=('r2', 'neg_mean_squared_error')
                   ,verbose=True)
train_val_=(scores['train_neg_mean_squared_error'])
test_val_=(scores['test_neg_mean_squared_error'])
train_val_r2=scores['train_r2']
test_val_r2=scores['test_r2']
train_val_RMSE=(abs(train_val_)**0.5)
test_val_RMSE=(abs(test_val_)**0.5)