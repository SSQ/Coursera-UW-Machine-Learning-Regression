
# Predicting house prices using k-nearest neighbors regression
In this notebook, you will implement k-nearest neighbors regression. You will:
  * Find the k-nearest neighbors of a given query input
  * Predict the output for the query input using the k-nearest neighbors
  * Choose the best value of k using a validation set

# Fire up GraphLab Create


```python
import graphlab
```

# Load in house sales data

For this notebook, we use a subset of the King County housing dataset created by randomly selecting 40% of the houses in the full dataset.


```python
sales = graphlab.SFrame('kc_house_data_small.gl/')
```

    [INFO] graphlab.cython.cy_server: GraphLab Create v2.1 started. Logging: C:\Users\SSQ\AppData\Local\Temp\graphlab_server_1481548213.log.0
    

    This non-commercial license of GraphLab Create for academic use is assigned to 2015200703@mail.buct.edu.cn and will expire on November 11, 2017.
    

# Import useful functions from previous notebooks

To efficiently compute pairwise distances among data points, we will convert the SFrame into a 2D Numpy array. First import the numpy library and then copy and paste `get_numpy_data()` from the second notebook of Week 2.


```python
import numpy as np # note this allows us to refer to numpy as np instead
```


```python
def get_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1 # this is how you add a constant column to an SFrame
    # add the column 'constant' to the front of the features list so that we can extract it along with the others:
    features = ['constant'] + features # this is how you combine two lists
    # select the columns of data_SFrame given by the features list into the SFrame features_sframe (now including constant):
    features_sframe = data_sframe[features]
    # the following line will convert the features_SFrame into a numpy matrix:
    feature_matrix = features_sframe.to_numpy()
    # assign the column of data_sframe associated with the output to the SArray output_sarray
    output_sarray = data_sframe[output]
    # the following will convert the SArray into a numpy array by first converting it to a list
    output_array = output_sarray.to_numpy()
    return(feature_matrix, output_array)
```

We will also need the `normalize_features()` function from Week 5 that normalizes all feature columns to unit norm. Paste this function below.


```python
def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix, axis=0)
    normalized_features = feature_matrix / norms
    return(normalized_features, norms)
```

# Split data into training, test, and validation sets


```python
(train_and_validation, test) = sales.random_split(.8, seed=1) # initial train/test split
(train, validation) = train_and_validation.random_split(.8, seed=1) # split training set into training and validation sets
```

# Extract features and normalize

Using all of the numerical inputs listed in `feature_list`, transform the training, test, and validation SFrames into Numpy arrays:


```python
feature_list = ['bedrooms',  
                'bathrooms',  
                'sqft_living',  
                'sqft_lot',  
                'floors',
                'waterfront',  
                'view',  
                'condition',  
                'grade',  
                'sqft_above',  
                'sqft_basement',
                'yr_built',  
                'yr_renovated',  
                'lat',  
                'long',  
                'sqft_living15',  
                'sqft_lot15']
features_train, output_train = get_numpy_data(train, feature_list, 'price')
features_test, output_test = get_numpy_data(test, feature_list, 'price')
features_valid, output_valid = get_numpy_data(validation, feature_list, 'price')
```

In computing distances, it is crucial to normalize features. Otherwise, for example, the `sqft_living` feature (typically on the order of thousands) would exert a much larger influence on distance than the `bedrooms` feature (typically on the order of ones). We divide each column of the training feature matrix by its 2-norm, so that the transformed column has unit norm.

IMPORTANT: Make sure to store the norms of the features in the training set. The features in the test and validation sets must be divided by these same norms, so that the training, test, and validation sets are normalized consistently.


```python
features_train, norms = normalize_features(features_train) # normalize training set features (columns)
features_test = features_test / norms # normalize test set by training set norms
features_valid = features_valid / norms # normalize validation set by training set norms
```

# Compute a single distance

To start, let's just explore computing the "distance" between two given houses.  We will take our **query house** to be the first house of the test set and look at the distance between this house and the 10th house of the training set.

To see the features associated with the query house, print the first row (index 0) of the test feature matrix. You should get an 18-dimensional vector whose components are between 0 and 1.


```python
query_feature = features_test[0]
print query_feature
```

    [ 0.01345102  0.01551285  0.01807473  0.01759212  0.00160518  0.017059    0.
      0.05102365  0.0116321   0.01564352  0.01362084  0.02481682  0.01350306
      0.          0.01345386 -0.01346927  0.01375926  0.0016225 ]
    

Now print the 10th row (index 9) of the training feature matrix. Again, you get an 18-dimensional vector with components between 0 and 1.


```python
features_train[0:10]
```




    array([[ 0.01345102,  0.01163464,  0.00602491,  0.00703685,  0.00181386,
             0.0085295 ,  0.        ,  0.        ,  0.0116321 ,  0.01216718,
             0.00811747,  0.        ,  0.01333931,  0.        ,  0.01343686,
            -0.01345621,  0.00861561,  0.00229178],
           [ 0.01345102,  0.01163464,  0.01355605,  0.01532602,  0.00232495,
             0.017059  ,  0.        ,  0.        ,  0.0116321 ,  0.01216718,
             0.01492789,  0.01023374,  0.01331201,  0.06650823,  0.01349619,
            -0.01346304,  0.01086596,  0.00309857],
           [ 0.01345102,  0.00775643,  0.00602491,  0.00459184,  0.00321036,
             0.0085295 ,  0.        ,  0.        ,  0.0116321 ,  0.01042901,
             0.00529699,  0.        ,  0.0131892 ,  0.        ,  0.01350097,
            -0.01345361,  0.0174884 ,  0.00327015],
           [ 0.01345102,  0.01551285,  0.01807473,  0.01168833,  0.00160518,
             0.0085295 ,  0.        ,  0.        ,  0.01938684,  0.01216718,
             0.00722317,  0.02328176,  0.01340754,  0.        ,  0.01343957,
            -0.01347122,  0.0087442 ,  0.00202813],
           [ 0.01345102,  0.01163464,  0.01204982,  0.01001856,  0.00259397,
             0.0085295 ,  0.        ,  0.        ,  0.0116321 ,  0.01390535,
             0.01155707,  0.        ,  0.01355765,  0.        ,  0.01346672,
            -0.01343289,  0.01157321,  0.00304341],
           [ 0.01345102,  0.01163464,  0.00903736,  0.00632124,  0.00311758,
             0.0085295 ,  0.        ,  0.        ,  0.0116321 ,  0.01216718,
             0.00729196,  0.        ,  0.01339389,  0.        ,  0.01340809,
            -0.01346257,  0.01060877,  0.00393903],
           [ 0.01345102,  0.01163464,  0.00602491,  0.01061491,  0.00239814,
             0.0085295 ,  0.        ,  0.        ,  0.0116321 ,  0.01216718,
             0.00722317,  0.01867657,  0.01337342,  0.        ,  0.01343716,
            -0.01346499,  0.01144462,  0.00329084],
           [ 0.01345102,  0.01163464,  0.01506227,  0.02122982,  0.00314487,
             0.0085295 ,  0.        ,  0.        ,  0.0116321 ,  0.01390535,
             0.01279533,  0.04349339,  0.01340754,  0.        ,  0.01346215,
            -0.01344394,  0.01420933,  0.0036202 ],
           [ 0.01345102,  0.01163464,  0.01054359,  0.01461041,  0.00086391,
             0.017059  ,  0.        ,  0.        ,  0.0116321 ,  0.01390535,
             0.01203862,  0.01790904,  0.01306638,  0.        ,  0.01347287,
            -0.01346755,  0.01131603,  0.0014493 ],
           [ 0.01345102,  0.01163464,  0.00602491,  0.0083488 ,  0.00050756,
             0.01279425,  0.        ,  0.        ,  0.01938684,  0.01390535,
             0.0096309 ,  0.        ,  0.01302544,  0.        ,  0.01346821,
            -0.01346254,  0.01195898,  0.00156612]])



***QUIZ QUESTION ***

What is the Euclidean distance between the query house and the 10th house of the training set? 

Note: Do not use the `np.linalg.norm` function; use `np.sqrt`, `np.sum`, and the power operator (`**`) instead. The latter approach is more easily adapted to computing multiple distances at once.


```python
squared_distance = (features_train - query_feature)**2
#print squared_distance
sum_distance = np.empty(10)
for i in range(10):
    sum_distance[i] = np.sum(squared_distance[i,:])
#print sum_distance
educlidean_distance = np.sqrt(sum_distance)
print educlidean_distance
```

    [ 0.06027471  0.08546881  0.06149946  0.05340274  0.05844484  0.05987922
      0.0546314   0.05543108  0.05238363  0.05972359]
    

# Compute multiple distances

Of course, to do nearest neighbor regression, we need to compute the distance between our query house and *all* houses in the training set.  

To visualize this nearest-neighbor search, let's first compute the distance from our query house (`features_test[0]`) to the first 10 houses of the training set (`features_train[0:10]`) and then search for the nearest neighbor within this small set of houses.  Through restricting ourselves to a small set of houses to begin with, we can visually scan the list of 10 distances to verify that our code for finding the nearest neighbor is working.

Write a loop to compute the Euclidean distance from the query house to each of the first 10 houses in the training set.


```python

```

*** QUIZ QUESTION ***

Among the first 10 training houses, which house is the closest to the query house?


```python

```

It is computationally inefficient to loop over computing distances to all houses in our training dataset. Fortunately, many of the Numpy functions can be **vectorized**, applying the same operation over multiple values or vectors.  We now walk through this process.

Consider the following loop that computes the element-wise difference between the features of the query house (`features_test[0]`) and the first 3 training houses (`features_train[0:3]`):


```python
for i in xrange(3):
    print features_train[i]-features_test[0]
    # should print 3 vectors of length 18
```

    [  0.00000000e+00  -3.87821276e-03  -1.20498190e-02  -1.05552733e-02
       2.08673616e-04  -8.52950206e-03   0.00000000e+00  -5.10236549e-02
       0.00000000e+00  -3.47633726e-03  -5.50336860e-03  -2.48168183e-02
      -1.63756198e-04   0.00000000e+00  -1.70072004e-05   1.30577772e-05
      -5.14364795e-03   6.69281453e-04]
    [  0.00000000e+00  -3.87821276e-03  -4.51868214e-03  -2.26610387e-03
       7.19763456e-04   0.00000000e+00   0.00000000e+00  -5.10236549e-02
       0.00000000e+00  -3.47633726e-03   1.30705004e-03  -1.45830788e-02
      -1.91048898e-04   6.65082271e-02   4.23240653e-05   6.22415897e-06
      -2.89330197e-03   1.47606982e-03]
    [  0.00000000e+00  -7.75642553e-03  -1.20498190e-02  -1.30002801e-02
       1.60518166e-03  -8.52950206e-03   0.00000000e+00  -5.10236549e-02
       0.00000000e+00  -5.21450589e-03  -8.32384500e-03  -2.48168183e-02
      -3.13866046e-04   0.00000000e+00   4.71047219e-05   1.56530415e-05
       3.72914476e-03   1.64764925e-03]
    

The subtraction operator (`-`) in Numpy is vectorized as follows:


```python
print features_train[0:3] - features_test[0]
```

    [[  0.00000000e+00  -3.87821276e-03  -1.20498190e-02  -1.05552733e-02
        2.08673616e-04  -8.52950206e-03   0.00000000e+00  -5.10236549e-02
        0.00000000e+00  -3.47633726e-03  -5.50336860e-03  -2.48168183e-02
       -1.63756198e-04   0.00000000e+00  -1.70072004e-05   1.30577772e-05
       -5.14364795e-03   6.69281453e-04]
     [  0.00000000e+00  -3.87821276e-03  -4.51868214e-03  -2.26610387e-03
        7.19763456e-04   0.00000000e+00   0.00000000e+00  -5.10236549e-02
        0.00000000e+00  -3.47633726e-03   1.30705004e-03  -1.45830788e-02
       -1.91048898e-04   6.65082271e-02   4.23240653e-05   6.22415897e-06
       -2.89330197e-03   1.47606982e-03]
     [  0.00000000e+00  -7.75642553e-03  -1.20498190e-02  -1.30002801e-02
        1.60518166e-03  -8.52950206e-03   0.00000000e+00  -5.10236549e-02
        0.00000000e+00  -5.21450589e-03  -8.32384500e-03  -2.48168183e-02
       -3.13866046e-04   0.00000000e+00   4.71047219e-05   1.56530415e-05
        3.72914476e-03   1.64764925e-03]]
    

Note that the output of this vectorized operation is identical to that of the loop above, which can be verified below:


```python
# verify that vectorization works
results = features_train[0:3] - features_test[0]
print results[0] - (features_train[0]-features_test[0])
# should print all 0's if results[0] == (features_train[0]-features_test[0])
print results[1] - (features_train[1]-features_test[0])
# should print all 0's if results[1] == (features_train[1]-features_test[0])
print results[2] - (features_train[2]-features_test[0])
# should print all 0's if results[2] == (features_train[2]-features_test[0])
```

    [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
    [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
    [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
    

Aside: it is a good idea to write tests like this cell whenever you are vectorizing a complicated operation.

# Perform 1-nearest neighbor regression

Now that we have the element-wise differences, it is not too hard to compute the Euclidean distances between our query house and all of the training houses. First, write a single-line expression to define a variable `diff` such that `diff[i]` gives the element-wise difference between the features of the query house and the `i`-th training house.


```python
diff = features_train - query_feature
```

To test the code above, run the following cell, which should output a value -0.0934339605842:


```python
print diff[-1].sum() # sum of the feature differences between the query and last training house
# should print -0.0934339605842
```

    -0.0934339605842
    

The next step in computing the Euclidean distances is to take these feature-by-feature differences in `diff`, square each, and take the sum over feature indices.  That is, compute the sum of square feature differences for each training house (row in `diff`).

By default, `np.sum` sums up everything in the matrix and returns a single number. To instead sum only over a row or column, we need to specifiy the `axis` parameter described in the `np.sum` [documentation](http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.sum.html). In particular, `axis=1` computes the sum across each row.

Below, we compute this sum of square feature differences for all training houses and verify that the output for the 16th house in the training set is equivalent to having examined only the 16th row of `diff` and computing the sum of squares on that row alone.


```python
print np.sum(diff**2, axis=1)[15] # take sum of squares across each row, and print the 16th sum
print np.sum(diff[15]**2) # print the sum of squares for the 16th row -- should be same as above
```

    0.00330705902879
    0.00330705902879
    

With this result in mind, write a single-line expression to compute the Euclidean distances between the query house and all houses in the training set. Assign the result to a variable `distances`.

**Hint**: Do not forget to take the square root of the sum of squares.


```python
distances = np.sqrt(np.sum(diff**2, axis=1))
```

To test the code above, run the following cell, which should output a value 0.0237082324496:


```python
print distances[100] # Euclidean distance between the query house and the 101th training house
# should print 0.0237082324496
```

    0.0237082324496
    

Now you are ready to write a function that computes the distances from a query house to all training houses. The function should take two parameters: (i) the matrix of training features and (ii) the single feature vector associated with the query.


```python
def compute_distance(training_features,query_feature):
    diff = training_features - query_feature
    distances = np.sqrt(np.sum(diff**2, axis=1))
    return(distances)
```

*** QUIZ QUESTIONS ***

1.  Take the query house to be third house of the test set (`features_test[2]`).  What is the index of the house in the training set that is closest to this query house?
2.  What is the predicted value of the query house based on 1-nearest neighbor regression?


```python
distance = compute_distance(features_train,features_test[2])
print np.argmin(distance)
```

    382
    


```python
print output_train[382]
```

    249000
    

# Perform k-nearest neighbor regression

For k-nearest neighbors, we need to find a *set* of k houses in the training set closest to a given query house. We then make predictions based on these k nearest neighbors.

## Fetch k-nearest neighbors

Using the functions above, implement a function that takes in
 * the value of k;
 * the feature matrix for the training houses; and
 * the feature vector of the query house
 
and returns the indices of the k closest training houses. For instance, with 2-nearest neighbor, a return value of [5, 10] would indicate that the 6th and 11th training houses are closest to the query house.

**Hint**: Look at the [documentation for `np.argsort`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html).


```python
def indices_of_KNN(k,features_train,query_feature):
    distance = compute_distance(features_train,query_feature)
    sorted_distance = np.argsort(distance)
    indice = sorted_distance[0:k]
    return(indice)
```

*** QUIZ QUESTION ***

Take the query house to be third house of the test set (`features_test[2]`).  What are the indices of the 4 training houses closest to the query house?


```python
indice = indices_of_KNN(4,features_train,features_test[2])
print indice
```

    [ 382 1149 4087 3142]
    

## Make a single prediction by averaging k nearest neighbor outputs

Now that we know how to find the k-nearest neighbors, write a function that predicts the value of a given query house. **For simplicity, take the average of the prices of the k nearest neighbors in the training set**. The function should have the following parameters:
 * the value of k;
 * the feature matrix for the training houses;
 * the output values (prices) of the training houses; and
 * the feature vector of the query house, whose price we are predicting.
 
The function should return a predicted value of the query house.

**Hint**: You can extract multiple items from a Numpy array using a list of indices. For instance, `output_train[[6, 10]]` returns the prices of the 7th and 11th training houses.


```python
def single_prediction_KNN(k,features_train,output_train,query_feature):
    indice = indices_of_KNN(k,features_train,query_feature)
    predicted_query = np.mean(output_train[indice])
    return(predicted_query)
```

*** QUIZ QUESTION ***

Again taking the query house to be third house of the test set (`features_test[2]`), predict the value of the query house using k-nearest neighbors with `k=4` and the simple averaging method described and implemented above.


```python
predicted_value = single_prediction_KNN(4,features_train,output_train,features_test[2])
print predicted_value
predicted_value = single_prediction_KNN(1,features_train,output_train,features_test[2])
print predicted_value
```

    413987.5
    249000.0
    

Compare this predicted value using 4-nearest neighbors to the predicted value using 1-nearest neighbor computed earlier.

## Make multiple predictions

Write a function to predict the value of *each and every* house in a query set. (The query set can be any subset of the dataset, be it the test set or validation set.) The idea is to have a loop where we take each house in the query set as the query house and make a prediction for that specific house. The new function should take the following parameters:
 * the value of k;
 * the feature matrix for the training houses;
 * the output values (prices) of the training houses; and
 * the feature matrix for the query set.
 
The function should return a set of predicted values, one for each house in the query set.

**Hint**: To get the number of houses in the query set, use the `.shape` field of the query features matrix. See [the documentation](http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.ndarray.shape.html).


```python
def multiple_prediction_KNN(k,features_train,output_train,query_feature_set):
    num_query_set = query_feature_set.shape[0]
    print num_query_set
    predicted_query = np.empty(num_query_set)
    for i in range(num_query_set):
        indice = indices_of_KNN(k,features_train,query_feature_set[i])
        predicted_query[i] = np.mean(output_train[indice])
    return(predicted_query)
```

*** QUIZ QUESTION ***

Make predictions for the first 10 houses in the test set using k-nearest neighbors with `k=10`. 

1. What is the index of the house in this query set that has the lowest predicted value? 
2. What is the predicted value of this house?


```python
predicted_values = multiple_prediction_KNN(10,features_train,output_train,features_test[0:10])
print predicted_values
```

    10
    [ 881300.   431860.   460595.   430200.   766750.   667420.   350032.
      512800.7  484000.   457235. ]
    

## Choosing the best value of k using a validation set

There remains a question of choosing the value of k to use in making predictions. Here, we use a validation set to choose this value. Write a loop that does the following:

* For `k` in [1, 2, ..., 15]:
    * Makes predictions for each house in the VALIDATION set using the k-nearest neighbors from the TRAINING set.
    * Computes the RSS for these predictions on the VALIDATION set
    * Stores the RSS computed above in `rss_all`
* Report which `k` produced the lowest RSS on VALIDATION set.

(Depending on your computing environment, this computation may take 10-15 minutes.)


```python
rss_all = np.empty(15)
for k in range(1,16):
    print k
    predicted_values = multiple_prediction_KNN(k,features_train,output_train,features_valid)
    error = predicted_values - output_valid
    rss_all[k-1] = np.dot(error,error)
print rss_all
```

    1
    1435
    2
    1435
    3
    1435
    4
    1435
    5
    1435
    6
    1435
    7
    1435
    8
    1435
    9
    1435
    10
    1435
    11
    1435
    12
    1435
    13
    1435
    14
    1435
    15
    1435
    [  1.05453830e+14   8.34450735e+13   7.26920960e+13   7.19467217e+13
       6.98465174e+13   6.88995444e+13   6.83419735e+13   6.73616787e+13
       6.83727280e+13   6.93350487e+13   6.95238552e+13   6.90499696e+13
       7.00112545e+13   7.09086989e+13   7.11069284e+13]
    

To visualize the performance as a function of `k`, plot the RSS on the VALIDATION set for each considered `k` value:


```python
import matplotlib.pyplot as plt
%matplotlib inline

kvals = range(1, 16)
plt.plot(kvals, rss_all,'bo-')
```




    [<matplotlib.lines.Line2D at 0x2044c208>]




![png](output_76_1.png)


***QUIZ QUESTION ***

What is the RSS on the TEST data using the value of k found above?  To be clear, sum over all houses in the TEST set.


```python
rss_all = np.empty(15)
for k in range(1,16):
    print k
    predicted_values = multiple_prediction_KNN(k,features_train,output_train,features_test)
    error = predicted_values - output_test
    rss_all[k-1] = np.dot(error,error)
print rss_all
```

    1
    1741
    2
    1741
    3
    1741
    4
    1741
    5
    1741
    6
    1741
    7
    1741
    8
    1741
    9
    1741
    10
    1741
    11
    1741
    12
    1741
    13
    1741
    14
    1741
    15
    1741
    [  1.89300603e+14   1.62969655e+14   1.49008587e+14   1.37914468e+14
       1.32270468e+14   1.32736446e+14   1.31757082e+14   1.33118824e+14
       1.33492572e+14   1.32558117e+14   1.32648119e+14   1.32270862e+14
       1.32766703e+14   1.33006256e+14   1.34342939e+14]
    


```python
rss_all[7]
```




    133118823551516.81




```python

```
