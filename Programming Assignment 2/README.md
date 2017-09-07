# Goal
- Predicting House Prices (Multiple Variables)
# File Description
- `.zip` files is data file.
  - [`kc_house_data.gl.zip`](https://github.com/SSQ/Coursera-UW-Machine-Learning-Regression/blob/master/Programming%20Assignment%201/kc_house_data.gl.zip) (unzip `kc_house_data.gl`) consists of 21,613 houses and 21 features
- `.ipynb` file is the solution of Week 2 program assignment 1
  - `week-2-multiple-regression-assignment-1.ipynb`
- `.html` file is the html version of `.ipynb` file.
  - `week-2-multiple-regression-assignment-1.html`
# Snapshot
open `.html` file via brower for quick look.
# Algorithm
- Multiple Regression: Linear regression with multiple features
# Implementation
- Use SFrames to do some feature engineering
- Use built-in GraphLab Create (or otherwise) functions to compute the regression weights (coefficients)
- Given the regression weights, predictors and outcome write a function to compute the Residual Sum of Squares
- Look at coefficients and interpret their meanings
- Evaluate multiple models via RSS
# Implementation in detail
- Load house sales data
- Split data into training and testing
- Add 4 new variables in both your train_data and test_data.
  - ‘bedrooms_squared’ = ‘bedrooms’*‘bedrooms’
  - ‘bed_bath_rooms’ = ‘bedrooms’*‘bathrooms’
  - ‘log_sqft_living’ = log(‘sqft_living’)
  - ‘lat_plus_long’ = ‘lat’ + ‘long’
- Estimate the regression coefficients/weights for predicting ‘price’ for the following three models:
  - Model 1: ‘sqft_living’, ‘bedrooms’, ‘bathrooms’, ‘lat’, and ‘long’
  - Model 2: ‘sqft_living’, ‘bedrooms’, ‘bathrooms’, ‘lat’,‘long’, and ‘bed_bath_rooms’
  - Model 3: ‘sqft_living’, ‘bedrooms’, ‘bathrooms’, ‘lat’,‘long’, ‘bed_bath_rooms’, ‘bedrooms_squared’, ‘log_sqft_living’, and ‘lat_plus_long’
- Predicting Values and calculate Residual Sum of Squares under these three model 
  - Training set: **Model 3** has the least RSS, Model 2 and Model 1
  - Testing set: **Model 2** has the least RSS, Model 1 and Model 3


