import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('./data/diabetes.csv')
# print(dataset.head())
print(dataset.shape)

zero_not_accepted = ['Glucose','BloodPressure','SkinThickness','BMI','Insulin']

#replace zeros and missing values
for column in zero_not_accepted:
    dataset[column] = dataset[column].replace(0,np.NaN)
    mean = dataset[column].mean(skipna = True)
    dataset[column] = dataset[column].replace(np.NaN, mean)

#split the dataset
X = dataset.iloc[:,0:8]
Y = dataset.iloc[:,8]

X_train ,X_test, y_train , y_test = train_test_split(X, Y, random_state=0, test_size=0.2)

#Feature Scaling
sc_X =StandardScaler()
X_train =sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

#Define the model, Fit and Predict
classifier = KNeighborsClassifier(n_neighbors=11,p=2,metric='euclidean')
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

#Evaluate the model
cm = confusion_matrix(y_test,y_pred)
print(cm)

print(f1_score(y_test,y_pred))

print(accuracy_score(y_test,y_pred))


