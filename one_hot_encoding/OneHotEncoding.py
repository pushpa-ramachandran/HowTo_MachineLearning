import pandas as pd
from sklearn import linear_model

data = pd.read_csv("NPS.csv")
print(data)

# get the one hot encoding for the sentiment data
dummy = pd.get_dummies(data.sentiment)

data_dummy = pd.concat([data,dummy], axis='columns')
print(data_dummy)

data_dummy.drop('sentiment',axis='columns',inplace= True )
print(data_dummy)

#drop one column to avoid dummy variable trap
data_dummy.drop('neutral', axis='columns',inplace = True)
print(data_dummy)

#Assign X and y - input and output variabels
X = data_dummy.drop(['score'],axis='columns')
y = data_dummy.score

#train the model
reg = linear_model.LinearRegression()
reg.fit(X,y)

pred = reg.predict(X)
print(pred)

#accuracy of the model
print(reg.score(X,y))

#predict for 8000 spend, positive sentiment
pred = reg.predict([[8000,0,1]])
print(pred)