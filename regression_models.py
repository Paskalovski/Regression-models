import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

## visualization of the data

def part1_scatter():
    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4)
    plt.show()
# part1_scatter()

## polynomial LinearRegression on training data for degrees 1,3,6,9
## returns 100 predicted values over the interval x=0 to 10

def regression_deg4():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    result = np.zeros((4, 100))
    poly_degree = [1, 3, 6, 9]
    for i, degree in enumerate(poly_degree):
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_train.reshape(len(X_train), 1))
        linreg = LinearRegression().fit(X_poly, y_train)
        test_data = np.linspace(0, 10, 100)
        y_test_data = linreg.predict(poly.fit_transform(test_data.reshape(len(test_data), 1)))
        result[i, :] = y_test_data
    return result


## visual presentation of the predicted models

def plot_one(degree_predictions):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10,5))
    plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
    plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
    for i,degree in enumerate([1,3,6,9]):
        plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
    plt.ylim(-1,2.5)
    plt.legend(loc=4)
    plt.show()
# plot_one(regression_deg4())


## regression score for the predicted models

def regression_score():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics.regression import r2_score

    R2_test = np.zeros(10)
    R2_train = np.zeros(10)
    for degree in range(10):
        #train polynomial linear regression
        poly = PolynomialFeatures(degree = degree)
        X_train_poly = poly.fit_transform(X_train.reshape(len(X_train),1))
        linreg = LinearRegression().fit(X_train_poly, y_train)
        #evaluate the polynomial linear regression
        X_test_poly = poly.fit_transform(X_test.reshape(len(X_test),1))
        y_test_pred = linreg.predict(X_test_poly)
        y_train_pred = linreg.predict(X_train_poly)
        R2_test[degree] = r2_score(y_test, y_test_pred)
        R2_train[degree] = r2_score(y_train, y_train_pred)

    return (R2_train, R2_test)


## Graphical presentation of the regression scores

def plot_regression_scores():
    import matplotlib.pyplot as plt
    R2_train, R2_test = regression_score()
    degrees = np.arange(0, 10)
    plt.figure()
    plt.plot(degrees, R2_train,'b', degrees, R2_test, 'r')
    plt.show()
# plot_regression_scores()


##regression scores for a non-regularized LinearRegression model and a lasso regression model

def regression_scores_linear_lasso():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics.regression import r2_score

    poly = PolynomialFeatures(degree=12)
    X_train_poly = poly.fit_transform(X_train.reshape(len(X_train), 1))
    linreg = LinearRegression().fit(X_train_poly, y_train)

    X_test_poly = poly.fit_transform(X_test.reshape(len(X_test), 1))
    y_test_pred = linreg.predict(X_test_poly)
    LinearRegression_R2_test_score = r2_score(y_test, y_test_pred)

    ##### Lasso regularized n=12 polynomial linear regression #####
    # train
    linreg = Lasso(alpha=0.01, max_iter=10000).fit(X_train_poly, y_train)
    # evaluate
    y_test_pred = linreg.predict(X_test_poly)
    Lasso_R2_test_score = r2_score(y_test, y_test_pred)

    return (LinearRegression_R2_test_score, Lasso_R2_test_score)

regression_scores_linear_lasso()

