#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Taking care of categorical variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()
#no need to enode the dependent variable

#avoiding the dummy vaiable trap
x = x[:, 1:]

#splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Fitting the multiple regression model to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Prediting the test set results
y_pred = regressor.predict(x_test)

#building the most optimal model using backward elimination
import statsmodels.api as sm
x = np.append(arr = np.ones((50, 1)).astype(int), values = x, axis = 1)
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()
#
#x_opt = x[:, [0, 1, 3, 4, 5]]
#regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
#regressor_ols.summary()
#
#x_opt = x[:, [0, 3, 4, 5]]
#regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
#regressor_ols.summary()
#
#x_opt = x[:, [0, 3, 5]]
#regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
#regressor_ols.summary()
#
#x_opt = x[:, [0, 3]]
#regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
#regressor_ols.summary()

def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = x[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)










