#https://github.com/codebasics/py/blob/master/ML/2_linear_reg_multivariate/2_linear_regression_multivariate.ipynb

import pandas as pd
import numpy as np
from sklearn import linear_model
import math

data = pd.read_csv('Prices.csv')
print(data)

#Data preprocessing
#Since few data are missing, find the median of the data and fill the missing data with the median
rooms_median = math.floor(data.rooms.median())
print(rooms_median)

#fill the NaN with the mediam values
data.rooms = data.rooms.fillna(rooms_median)
print(data)

#train the linear regression model
reg = linear_model.LinearRegression()
reg.fit(data[['sqft','rooms','years']],data.price)

#predict with the new set of values
price_predicted = reg.predict([[1500,3,0]])

print(price_predicted)

