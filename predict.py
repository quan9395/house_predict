import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

#read data
data = pd.read_csv('data/train.csv', index_col="Id")

#choose features
features = ["LotArea","YearBuilt","1stFlrSF","2ndFlrSF","FullBath","BedroomAbvGr","TotRmsAbvGrd"]
X = data[features]
y = data["SalePrice"]

#split data into X_train, y_train, X_valid, y_valid
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

#train model with Decision Tree
dt_model = DecisionTreeRegressor(random_state=1)
dt_model.fit(X_train, y_train)

y_predicts_dt = dt_model.predict(X_valid)
print("This is y_predicts from Decision Tree model:")
print(pd.DataFrame({'y':y_valid, 'y_predicts': y_predicts_dt}))


#train model with Random Forest
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(X_train, y_train)

y_predicts_rf = rf_model.predict(X_valid)
print("This is y_predicts from Random Forest model:")
print(pd.DataFrame({'y':y_valid, 'y_predicts': y_predicts_rf}))

#Predict with 1 input
print(rf_model.predict([[30000,1960,2000,0,4,3,2]]))