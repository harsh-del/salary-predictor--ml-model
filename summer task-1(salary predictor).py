import pandas as pd
dataset=pd.read_csv('Salary.csv')
dataset.columns
dataset
x = dataset['YearsExperience'].values.reshape(-1,1)
type(x)
y = dataset['Salary']
type(y)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
model.fit(X_train,y_train)
y_test
model.predict(X_test)
y_test
import joblib
joblib.dump(model,'salary_predict.pk1')

