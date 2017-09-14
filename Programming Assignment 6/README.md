# Goal
Implement ridge regression via gradient descent
# File Description
- `.zip` files is data file.
  - [`kc_house_data.gl.zip`](https://github.com/SSQ/Coursera-UW-Machine-Learning-Regression/blob/master/Programming%20Assignment%201/kc_house_data.gl.zip) (unzip `kc_house_data.gl`) consists of 21,613 houses and 21 features
- `.ipynb` file is the solution of Week 4 program assignment 6
  - `week-4-ridge-regression-assignment-2.ipynb`
- `.html` file is the html version of `.ipynb` file.
  - `week-4-ridge-regression-assignment-2.html`
# Snapshot
open `.html` file via brower for quick look.
# Algorithm
- ridge regression with gradient descent
# Implementation
- Convert an SFrame into a Numpy array (if applicable)
- Write a Numpy function to compute the derivative of the regression weights with respect to a single feature
- Write gradient descent function to compute the regression weights given an initial weight vector, step size, tolerance, and L2 penalty
# Implementation in detail
- Write a function `get_numpy_data(data_sframe, features, output)` convert the SFrame into a 2D Numpy array
- Write a function `feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant)` return the derivative for the weight for feature i 
- Write a function `ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations=100)` return weights
- `sqft_living` with 0(**lower RSS**) and 1e11 L2 regularization
- `['sqft_living', 'sqft_living15']` with 0(**lower RSS**) and 1e11 L2 regularization 

