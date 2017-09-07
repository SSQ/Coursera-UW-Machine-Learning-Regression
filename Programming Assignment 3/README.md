# Goal
Estimating Multiple Regression Coefficients (Gradient Descent)
# File Description
- `.zip` files is data file.
  - [`kc_house_data.gl.zip`](https://github.com/SSQ/Coursera-UW-Machine-Learning-Regression/blob/master/Programming%20Assignment%201/kc_house_data.gl.zip) (unzip `kc_house_data.gl`) consists of 21,613 houses and 21 features
- `.ipynb` file is the solution of Week 2 program assignment 2
  - `week-2-multiple-regression-assignment-2.ipynb`
- `.html` file is the html version of `.ipynb` file.
  - `week-2-multiple-regression-assignment-2.html`
# Snapshot
open `.html` file via brower for quick look.
# Algorithm
- Estimating Multiple Regression Coefficients (Gradient Descent)
# Implementation
- Add a constant column of 1's to a SFrame (or otherwise) to account for the intercept
- Convert an SFrame into a numpy array
- Write a predict_output() function using numpy
- Write a numpy function to compute the derivative of the regression weights with respect to a single feature
- Write gradient descent function to compute the regression weights given an initial weight vector, step size and tolerance.
- Use the gradient descent function to estimate regression weights for multiple features
# Implementation in detail
- Load house sales data
- Split data into training and testing
- Write a get_numpy_data(data_sframe, features, output) to convert data to numpy
- Write predict_outcome(feature_matrix, weights)
- Write feature_derivative(errors, feature)
- Impletement regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance)
- Run the regression_gradient_descent function to estimate the model 1 (features: ‘sqft_living’)
- Run the regression_gradient_descent function to estimate the model 2 (features: 'sqft_living', 'sqft_living15')
- **Model 2** has less RSS in test data.


