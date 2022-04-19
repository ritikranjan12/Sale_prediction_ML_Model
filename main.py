import pandas as pd # useful for loading dataset
import numpy as np # to perform array
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error

dataset = pd.read_csv(r'./advertising.csv')


x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

clf = LinearRegression()
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

tv = float(input("Enter the advertisement done by TV : " ))
rad = float(input("Enter the advertisement done by Radio : " ))
news = float(input("Enter the advertisement done by Newspaper : " ))
newinp = [[tv,rad,news]]
pred = clf.predict(newinp)
print(pred)
acc = mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred)
print("Accuracy of our Prediction is ",(1-acc)*100)