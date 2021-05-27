import pandas as pd
dataset=pd.read_csv('Salary.csv')
x = dataset['YearsExperience'].values.reshape(-1,1)
y = dataset['Salary']
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)
years=input("Enter experience of an indiviual(in years):")
print("Salary:",model.predict([[float(years)]]))


