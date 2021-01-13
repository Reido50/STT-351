import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import statsmodels.api as sm
import pandas as pd
from sklearn.linear_model import LinearRegression

def single_reg(target, explanatory):
    # y = target, x = explanatory variable
    x=np.array(explanatory)
    y=np.array(target)
    
    # Make the regression
    model = LinearRegression(fit_intercept=True)
    model.fit(x[:, np.newaxis], y)
    
    # Display data and regression
    xfit = np.linspace(0, 3, 1000)
    yfit = model.predict(xfit[:, np.newaxis])
    plt.scatter(x, y)
    plt.plot(xfit, yfit);
    
    # Show numerical results of regression
    X2 = sm.add_constant(x)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    print(est2.summary())

def multi_reg(target, explanatory1, explanatory2):
    # y = target, x1 = explanatory1, x2 = explanatory2
    x1=np.array(explanatory1)
    x2=np.array(explanatory2)
    y=np.array(target)
    x=np.column_stack((x1,x2))
    
    # Make the regression
    model = LinearRegression(fit_intercept=True)
    model.fit(x, y)
    
    print(model.coef_)

    X5 = sm.add_constant(x)
    est = sm.OLS(y, X5)
    est2 = est.fit()
    print(est2.summary())
    
    xfit=np.column_stack((x1,x2))
    yfit=model.predict(xfit)
    resid=y-yfit
    stats.probplot(resid,plot=plt)
    
def single_log_reg(target, explanatory):
    # y = target, x = explanatory variable
    x=np.array(explanatory)
    y=np.array(target)
    x=np.log(x)
    y=np.log(y)
    
    # Make the regression
    model = LinearRegression(fit_intercept=True)
    model.fit(x[:, np.newaxis], y)
    
    # Display data and regression
    xfit = np.linspace(0, 3, 1000)
    yfit = model.predict(xfit[:, np.newaxis])
    plt.scatter(x, y)
    plt.plot(xfit, yfit);
    
    # Show numerical results of regression
    X2 = sm.add_constant(x)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    print(est2.summary())
    
def three_reg(target, explanatory1, explanatory2, explanatory3):
    # y = target, x1 = explanatory1, x2 = explanatory2, x3 = explanatory3
    x1=np.array(explanatory1)
    x2=np.array(explanatory2)
    x3=np.array(explanatory3)
    y=np.array(target)
    x=np.column_stack((x1,x2,x3))
    
    # Make the regression
    model = LinearRegression(fit_intercept=True)
    model.fit(x, y)
    
    print(model.coef_)

    X5 = sm.add_constant(x)
    est = sm.OLS(y, X5)
    est2 = est.fit()
    print(est2.summary())
    
    xfit=np.column_stack((x1,x2,x3))
    yfit=model.predict(xfit)
    resid=y-yfit
    stats.probplot(resid,plot=plt)
    

# Read in the data set
iris=pd.read_csv("iris.csv",sep=',')

#
# General single regressions
#
print("*\n*\n*\n*")
print("GENERAL SINGLE REGRESSIONS");
# sepal length vs sepal width
single_reg(iris.sepal_length, iris.sepal_width)
# sepal length vs petal length
single_reg(iris.sepal_length, iris.petal_length)
# sepal length vs petal width
single_reg(iris.sepal_length, iris.petal_width)
# Show and then clear the plot
plt.show()
plt.clf()

# Split by variety
split_arr = np.array_split(iris, 3)

#
# Setosa single regressions
#
print("*\n*\n*\n*")
print("SETOSA SINGLE REGRESSIONS")
# sepal length vs sepal width
single_reg(split_arr[0].sepal_length, split_arr[0].sepal_width)
# sepal length vs petal length
single_reg(split_arr[0].sepal_length, split_arr[0].petal_length)
# sepal length vs petal width
single_reg(split_arr[0].sepal_length, split_arr[0].petal_width)
# Show and then clear the plot
plt.show()
plt.clf()

#
# Versicolor single regressions
#
print("*\n*\n*\n*")
print("VERSICOLOR SINGLE REGRESSIONS")
# sepal length vs sepal width
single_reg(split_arr[1].sepal_length, split_arr[1].sepal_width)
# sepal length vs petal length
single_reg(split_arr[1].sepal_length, split_arr[1].petal_length)
# sepal length vs petal width
single_reg(split_arr[1].sepal_length, split_arr[1].petal_width)
# Show and then clear the plot
plt.show()
plt.clf()

#
# Virginica single regressions
#
print("*\n*\n*\n*")
print("VIRGINICA SINGLE REGRESSIONS")
# sepal length vs sepal width
single_reg(split_arr[2].sepal_length, split_arr[2].sepal_width)
# sepal length vs petal length
single_reg(split_arr[2].sepal_length, split_arr[2].petal_length)
# sepal length vs petal width
single_reg(split_arr[2].sepal_length, split_arr[2].petal_width)
# Show and then clear the plot
plt.show()
plt.clf()

#
# General multiple regressions
#
print("*\n*\n*\n*")
print("GENERAL MULTIPLE REGRESSIONS")
# sepal length vs sepal width & petal width
multi_reg(iris.sepal_length, iris.sepal_width, iris.petal_width)
# Show and then clear the plot
plt.show()
plt.clf()
# sepal length vs sepal width & petal length
multi_reg(iris.sepal_length, iris.sepal_width, iris.petal_length)
# Show and then clear the plot
plt.show()
plt.clf()
# sepal length vs petal width & petal length
multi_reg(iris.sepal_length, iris.petal_width, iris.petal_length)
# Show and then clear the plot
plt.show()
plt.clf()
# sepal length vs all 3
three_reg(iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width)
# Show and then clear the plot
plt.show()
plt.clf()

#
# Setosa multiple regressions
#
print("*\n*\n*\n*")
print("SETOSA MULTIPLE REGRESSIONS")
# sepal length vs sepal width & petal width
multi_reg(split_arr[0].sepal_length, split_arr[0].sepal_width, split_arr[0].petal_width)
# Show and then clear the plot
plt.show()
plt.clf()
# sepal length vs sepal width & petal length
multi_reg(split_arr[0].sepal_length, split_arr[0].sepal_width, split_arr[0].petal_length)
# Show and then clear the plot
plt.show()
plt.clf()
# sepal length vs petal width & petal length
multi_reg(split_arr[0].sepal_length, split_arr[0].petal_width, split_arr[0].petal_length)
# Show and then clear the plot
plt.show()
plt.clf()
# sepal length vs all 3
three_reg(split_arr[0].sepal_length, split_arr[0].sepal_width, split_arr[0].petal_length, split_arr[0].petal_width)
# Show and then clear the plot
plt.show()
plt.clf()

#
# Versicolor multiple regressions
#
print("*\n*\n*\n*")
print("VERSICOLOR MULTIPLE REGRESSIONS")
# sepal length vs sepal width & petal width
multi_reg(split_arr[1].sepal_length, split_arr[1].sepal_width, split_arr[1].petal_width)
# Show and then clear the plot
plt.show()
plt.clf()
# sepal length vs sepal width & petal length
multi_reg(split_arr[1].sepal_length, split_arr[1].sepal_width, split_arr[1].petal_length)
# Show and then clear the plot
plt.show()
plt.clf()
# sepal length vs petal width & petal length
multi_reg(split_arr[1].sepal_length, split_arr[1].petal_width, split_arr[1].petal_length)
# Show and then clear the plot
plt.show()
plt.clf()
# sepal length vs all 3
three_reg(split_arr[1].sepal_length, split_arr[1].sepal_width, split_arr[1].petal_length, split_arr[1].petal_width)
# Show and then clear the plot
plt.show()
plt.clf()

#
# Virginica multiple regressions
#
print("*\n*\n*\n*")
print("VIRGINICA MULTPLE REGRESSIONS")
# sepal length vs sepal width & petal width
multi_reg(split_arr[2].sepal_length, split_arr[2].sepal_width, split_arr[2].petal_width)
# Show and then clear the plot
plt.show()
plt.clf()
# sepal length vs sepal width & petal length
multi_reg(split_arr[2].sepal_length, split_arr[2].sepal_width, split_arr[2].petal_length)
# Show and then clear the plot
plt.show()
plt.clf()
# sepal length vs petal width & petal length
multi_reg(split_arr[2].sepal_length, split_arr[2].petal_width, split_arr[2].petal_length)
# Show and then clear the plot
plt.show()
plt.clf()
# sepal length vs all 3
three_reg(split_arr[2].sepal_length, split_arr[2].sepal_width, split_arr[2].petal_length, split_arr[2].petal_width)
# Show and then clear the plot
plt.show()
plt.clf()

#
# Power Regressions
#
print("*\n*\n*\n*")
print("SEPOSA POWER REGRESSION")
# sepal length vs sepal width
single_log_reg(split_arr[0].sepal_length, split_arr[0].sepal_width)
print("*\n*\n*\n*")
print("VERSICOLOR POWER REGRESSION")
# sepal length vs petal length
single_log_reg(split_arr[1].sepal_length, split_arr[1].petal_length)
print("*\n*\n*\n*")
print("VIRGINICA POWER REGRESSION")
# sepal length vs petal length
single_log_reg(split_arr[2].sepal_length, split_arr[2].petal_length)

