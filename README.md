# Classification-and-Regression

Problem 1:
Implement Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA). Refer
to part B.3 slides and handouts. Implement two functions in Python: ldaLearn and qdaLearn which take
a training data set (a feature matrix and labels) and return the means and covariance matrix (or matrices).
Implement two functions ldaTest and qdaTest which return the true labels for a given test data set and
the accuracy using the true labels for the test data.

Problem 2:
Implement ordinary least squares method to estimate regression parameters by minimizing the squared loss.
Implement the function learnOLERegression. Also implement the function testOLERegression to apply the learnt
weights for prediction on both training and testing data and to calculate the root mean squared error

Problem 3:
Implement parameter estimation for ridge regression by minimizing the regularized squared loss.
Implement it in the function learnRidgeRegression.

Problem 4:
Regression parameters can be calculated directly using analytical expressions (as in
Problem 2 and 3). To avoid computation of (X>X)􀀀1, another option is to use gradient descent to minimize
the loss function (or to maximize the log-likelihood) function. In this problem, you have to implement the
gradient descent procedure for estimating the weights w.
Use the minimize function (from the scipy library) which is same as the minimizer. Implement a function regressionObjVal to compute the regularized squared error and its gradient

Problem 5:
In this problem we will investigate the impact of using higher order polynomials for the input features. For
this problem use the third variable as the only input variable:
x t r a i n = x t r a i n [ : , 3 ]
x t e s t = x t e s t [ : , 3 ]
Implement the function mapNonLinear.m which converts a single attribute x into a vector of p attributes,
1; x; x2; : : : ; xp.

Problem 6:
Using the results obtained for previous 4 problems, make final recommendations for anyone using regression
for predicting diabetes level using the input features.
