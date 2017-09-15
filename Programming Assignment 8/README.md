# Goal
Implement your very own LASSO solver via coordinate descent.
# File Description
- `.zip` files is data file.
  - [`kc_house_data.gl.zip`](https://github.com/SSQ/Coursera-UW-Machine-Learning-Regression/blob/master/Programming%20Assignment%201/kc_house_data.gl.zip) (unzip `kc_house_data.gl`) consists of 21,613 houses and 21 features
- `.ipynb` file is the solution of Week 5 program assignment 8
  - `week-5-lasso-assignment-2.html`
- `.html` file is the html version of `.ipynb` file.
  - `week-5-lasso-assignment-2.html`
# Snapshot
open `.html` file via brower for quick look.
# Algorithm
- LASSO via coordinate descent.
# Implementation
- Write a function to normalize features
- Implement coordinate descent for LASSO
- Explore effects of L1 penalty
# Implementation in detail
- Write a function `normalize_features(feature_matrix)` `return(normalized_features, norms)`
- Write a function `lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)` `return new_weight_i` for Single Coordinate Descent Step
- Write a function `lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance)` `return weights` for Cyclical coordinate descent
- Evaluating LASSO fit with 13 features and 2 L1 penalty**(1e4, 1e7, 1e8)**
- Evaluating each of the learned models on the test data, **1e4** has the lowest RSS

