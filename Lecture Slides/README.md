Week 1:
- 1.0 Introduction
- 1.1 Simple Regression: Linear regression with one input
  - Describe the input (features) and output (real-valued predictions) of a regression model
  - Calculate a goodness-of-fit metric (e.g., RSS)
  - Estimate model parameters to minimize RSS using gradient descent
  - Interpret estimated model parameters
  - Exploit the estimated model to form predictions
  - Discuss the possible influence of high leverage points
  - Describe intuitively how fitted line might change when assuming different goodness-of-fit metrics

Week 2:
- 2.0: Multiple Regression: Linear regression with multiple features
  - Describe polynomial regression
  - Detrend a time series using trend and seasonal components
  - Write a regression model using multiple inputs or features thereof
  - Cast both polynomial regression and regression with multiple inputs as regression with multiple features
  - Calculate a goodness-of-fit metric (e.g., RSS)
  - Estimate model parameters of a general multiple regression model to minimize RSS:
    - In closed form
    - Using an iterative gradient descent algorithm
  - Interpret the coefficients of a non-featurized multiple regression fit
  - Exploit the estimated model to form predictions
  - Explain applications of multiple regression beyond house price modeling

Week 3:
- 3.0: Assessing Performance
  - Describe what a loss function is and give examples
  - Contrast training, generalization, and test error
  - Compute training and test error given a loss function
  - Discuss issue of assessing performance on training set
  - Describe tradeoffs in forming training/test splits
  - List and interpret the 3 sources of avg. prediction error
    - Irreducible error, bias, and variance
  - Discuss issue of selecting model complexity on test data and then using test error to assess generalization error
  - Motivate use of a validation set for selecting tuning parameters (e.g., model complexity)
  - Describe overall regression workflow
