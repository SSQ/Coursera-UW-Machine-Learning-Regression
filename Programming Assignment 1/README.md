# Goal
- Predicting House Prices (One feature)
# File Description
- `.zip` files is data file.
  - `kc_house_data.gl.zip` (unzip `kc_house_data.gl`) consists of 21,613 houses and 21 features
- `.ipynb` file is the solution of Week 1 program assignment
  - `week-1-simple-regression-assignment.ipynb`
- `.html` file is the html version of `.ipynb` file.
  - `week-1-simple-regression-assignment.html`
# Snapshot
open `.html` file via brower for quick look.
# Algorithm
- Simple Linear Regression
# Implementation
- Use SArray and SFrame functions to compute important summary statistics
- Write a function to compute the Simple Linear Regression weights using the closed form solution
- Write a function to make predictions of the output given the input feature
- Turn the regression around to predict the input/feature given the output
- Compare two different models for predicting house prices
# Implementation in detail
- Load house sales data
- Split data into training and testing
- Build a generic simple linear regression function
- Predicting Values
- Residual Sum of Squares
- Predict the squarefeet given price
- New Model: estimate prices from bedrooms
- Test your Linear Regression Algorithm on two models.(compared to bedrooms, sqrt feet results in less root of sum square)

