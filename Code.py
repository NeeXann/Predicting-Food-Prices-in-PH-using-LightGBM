#importing necessary libraries
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, cross_val_predict
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn import utils
import category_encoders as ce
gc.enable()

#importing and splitting the data into training and testing sets
df = pd.read_csv('https://raw.githubusercontent.com/NeeXann/Predicting-Food-Prices-in-PH-using-LightGBM/main/wfp_food_prices_phl.csv')

#Removing the index rows
df = df.drop(labels=[0], axis=0)
#extracting the feature variables and making their type as category
features = df[['market','category', 'commodity']]
features = features.astype('category')
#extrecting the price and converting its type into float64
price = df[['price']]
price = price.astype(float)

#Creating the train test dataset (splitting by 50%)
data_train, data_test, price_train, price_test = train_test_split(features,price, test_size=0.5, random_state=14, stratify=features[['commodity']])



#preparing the train data
X_train, X_test, y_train, y_test = train_test_split(data_train, price_train, test_size = 0.2, random_state = 14, stratify=data_train[['commodity']])

#lightgbm model
clf = lgb.LGBMRegressor(num_leaves= 14, max_depth = 17, 
                         random_state = 314, 
                         silent = True,
                         objective = 'regression',
                         categorical_feature = ['market', 'commodity'],
                         metric = 'auc', 
                         device_type = 'cpu',
                         n_estimators = 1000,
                         colsample_bytree = 0.9,
                         subsample = 0.8,
                         bagging_freq = 7,
                         learning_rate = 0.1, 
                       )

clf.fit(X_train, y_train, eval_set = [(X_test,y_test)],eval_metric = 'l1', early_stopping_rounds = 1000)


#Testing the accuracy of the created model
print('Training accuracy: {:.4f}'.format(clf.score(X_train,y_train)))
print('Training accuracy: {:.4f}'.format(clf.score(X_test,y_test)))
#Testing the accuracy of the model on the untouched data to measure the overfitting
print('Training accuracy: {:.4f}'.format(clf.score(data_test,price_test)))

#Creating the predicted values of the model and generating the scatterplot between the predicted price and the actual price of the foods.
y_pred = clf.predict(X_train)
y_pred_t = clf.predict(data_test)
mat=plt.scatter(y_train, y_pred)
mat_test=plt.scatter(price_test, y_pred_t)

lgb.plot_importance(clf)