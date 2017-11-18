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

Week 4:
- 4.0: Ridge Regression
  - Describe what happens to magnitude of estimated coefficients when model is overfit
  - Motivate form of ridge regression cost function
  - Describe what happens to estimated coefficients of ridge regression as tuning parameter λ is varied
  - Interpret coefficient path plot
  - Estimate ridge regression parameters:
    - In closed form
    - Using an iterative gradient descent algorithm
  - Implement K-fold cross validation to select the ridge regression tuning parameter λ
  
Week 5:
- 5.0: Lasso Regression: Regularization for feature selection
    - Perform feature selection using “all subsets” and “forward stepwise” algorithms
    - Analyze computational costs of these algorithms
    - Contrast greedy and optimal algorithms
    - Formulate lasso objective
    - Describe what happens to estimated lasso coefficients as tuning parameter λ is varied
    - Interpret lasso coefficient path plot
    - Contrast ridge and lasso regression
    - Describe geometrically why L1 penalty leads to sparsity
    - Estimate lasso regression parameters using an iterative coordinate descent algorithm
    - Implement K-fold cross validation to select lasso tuning parameter λ

Week 6: 
- 6.0: Going nonparametric: Nearest neighbor and kernel regression
  - Motivating local fits
    - Limitations of parametric regression
  - Nearest neighbor regression
    - 1-Nearest neighbor regression approach
    - Distance metrics
    - 1-Nearest neighbor algorithm
  - k-Nearest neighbors and weighted k-nearest neighbors
    - k-Nearest neighbors regression
    - k-Nearest neighbors in practice
    - Weighted k-nearest neighbors
  - Kernel regression
    - From weighted k-NN to kernel regression
    - Global fits of parametric models vs. local fits of kernel regression
  - k-NN and kernel regression wrapup
    - Performance of NN as amount of data grows
    - Issues with high-dimensions, data scarcity, and computational complexity
    - k-NN for classification
    - A brief recap
- 6.1: Recap & Look ahead
  - What we've learned
    - Simple and multiple regression
    - Assessing performance and ridge regression
    - Feature selection, lasso, and nearest neighbor regression
  - Summary and what's ahead in the specialization
    - What we covered and what we didn't cover
    - Thank you!
