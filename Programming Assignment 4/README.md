# Goal
Compare different regression models in order to assess which model fits best
# File Description
- `.zip` files is data file.
  - [`kc_house_data.gl.zip`](https://github.com/SSQ/Coursera-UW-Machine-Learning-Regression/blob/master/Programming%20Assignment%201/kc_house_data.gl.zip) (unzip `kc_house_data.gl`) consists of 21,613 houses and 21 features
- `.ipynb` file is the solution of Week 3 program assignment 4
  - `week-3-polynomial-regression-assignment.ipynb`
- `.html` file is the html version of `.ipynb` file.
  - `week-3-polynomial-regression-assignment.html`
# Snapshot
open `.html` file via brower for quick look.
# Algorithm
- Cross validation
# Implementation
- Write a function to take an an array and a degree and return an data frame where each column is the array to a polynomial value up to the total degree.
- Use a plotting tool (e.g. matplotlib) to visualize polynomial regressions
- Use a plotting tool (e.g. matplotlib) to visualize the same polynomial degree on different subsets of the data
- Use a validation set to select a polynomial degree
- Assess the final fit using test data
# Implementation in detail
- Write a function `polynomial_sframe(feature, degree)` to to create an SFrame consisting of the powers of an SArray up to a specific degree
- Visualizing polynomial regression with 
  - model1 = graphlab.linear_regression.create(poly1_data, target = 'price', features = ['power_1'], validation_set = None)
  - model2 = graphlab.linear_regression.create(poly2_data, target = 'price', features = my_features, validation_set = None)
  - model3 = graphlab.linear_regression.create(poly3_data, target = 'price', features = my_features, validation_set = None)
  - model15 = graphlab.linear_regression.create(poly15_data, target = 'price', features = my_features, validation_set = None)
- Visualizing 15 order polynomial regression with 4 data sets
- Write a loop for training fifteen 1 - 15 order models in training set and seeking for the least RSS in validation set to identify the property order **6 order**
- Test in the test data with property **6 order**
