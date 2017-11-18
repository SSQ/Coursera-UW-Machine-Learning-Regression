# Predicting house prices using k-nearest neighbors regression

# Goal
- Find the k-nearest neighbors of a given query input
- Predict the output for the query input using the k-nearest neighbors
- Choose the best value of k using a validation set

# File Description
- `.zip` files is data file.
  - [`kc_house_data_small.gl.zip`]() (unzip `kc_house_data_small.gl`) consists of approximatly 8k houses and 21 features
- description files
  - `.ipynb` file is the solution of Week 6 program assignment 1
    - `week-6-local-regression-assignment-blank.ipynb`
  - `.html` file is the html version of `.ipynb` file.
    - `week-6-local-regression-assignment-blank.html`
  - `.py`
    - `week-6-local-regression-assignment-blank.py`
  - file
    - week-6-local-regression-assignment-blank
    
# Snapshot
- **Recommend** open `md` file inside a file
- open `.html` file via brower for quick look.

# Algorithm
- k-nearest neighbors

# Implementation in detail
- Load in house sales data
- Import useful functions from previous notebooks
- Split data into training, test, and validation sets
- Extract features and normalize
- Compute a single distance
- Compute multiple distances
- Perform 1-nearest neighbor regression
- Perform k-nearest neighbor regression
  - Fetch k-nearest neighbors
    - the value of k;
    - the feature matrix for the instances; and
    - the feature of the query
  - Make a single prediction by averaging k nearest neighbor outputs
    - the value of k;
    - the feature matrix for the training houses;
    - the output values (prices) of the training houses; and
    - the feature vector of the query house, whose price we are predicting.
  - Make multiple predictions
    - the value of k;
    - the feature matrix for the training houses;
    - the output values (prices) of the training houses; and
    - the feature matrix for the query set.  
  - Choosing the best value of k using a validation set
