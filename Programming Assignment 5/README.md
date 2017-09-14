# Goal
Run ridge regression multiple times with different L2 penalties to see which one produces the best fit
# File Description
- `.zip` files is data file.
  - [`kc_house_data.gl.zip`](https://github.com/SSQ/Coursera-UW-Machine-Learning-Regression/blob/master/Programming%20Assignment%201/kc_house_data.gl.zip) (unzip `kc_house_data.gl`) consists of 21,613 houses and 21 features
- `.ipynb` file is the solution of Week 4 program assignment 5
  - `week-4-ridge-regression-assignment-1.ipynb`
- `.html` file is the html version of `.ipynb` file.
  - `week-4-ridge-regression-assignment-1.html`
# Snapshot
open `.html` file via brower for quick look.
# Algorithm
- ridge regression
# Implementation
- Use a pre-built implementation of regression to run polynomial regression
- Use matplotlib to visualize polynomial regressions
- Use a pre-built implementation of regression to run polynomial regression, this time with L2 penalty
- Use matplotlib to visualize polynomial regressions under L2 regularization
- Choose best L2 penalty using cross-validation.
- Assess the final fit using test data.
# Implementation in detail
- Recall function `polynomial_sframe(feature, degree)` from last Programming Assignment
- Visualize 2 L2 penalty (1e5 and 1e-5) on 4 data sets with pre-built function `graphlab.linear_regression.create(poly15_data,target='price',features=my_features,l2_penalty=l2_small_penalty,validation_set=None)`
- Write a function `k_fold_cross_validation(k, l2_penalty, data, output_name, features_list)` return average k fold validation error 
- using the `sqft_living` input, `l2_penalty in [10^1, 10^1.5, 10^2, 10^2.5, ..., 10^7]` to select best value for the L2 penalty **1e3**
- Visualize k-fold cross-validation errors for the L2 penalty

