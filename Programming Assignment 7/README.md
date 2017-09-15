# Goal
Use LASSO to select features, building on a pre-implemented solver for LASSO
# File Description
- `.zip` files is data file.
  - [`kc_house_data.gl.zip`](https://github.com/SSQ/Coursera-UW-Machine-Learning-Regression/blob/master/Programming%20Assignment%201/kc_house_data.gl.zip) (unzip `kc_house_data.gl`) consists of 21,613 houses and 21 features
- `.ipynb` file is the solution of Week 5 program assignment 7
  - `week-5-lasso-assignment-1.ipynb`
- `.html` file is the html version of `.ipynb` file.
  - `week-5-lasso-assignment-1.html`
# Snapshot
open `.html` file via brower for quick look.
# Algorithm
- LASSO
# Implementation
- Run LASSO with different L1 penalties.
- Choose best L1 penalty using a validation set.
- Choose best L1 penalty using a validation set, with additional constraint on the size of subset.
# Implementation in detail
- Use pre-built function and validation set to select the best L1 penalty with the lowest RSS in 10 L1 penaltys
- Limit the number of nonzero weights and select the lowest RSS on the validation set

