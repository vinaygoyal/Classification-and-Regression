import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys
import scipy.linalg as linalg
%matplotlib inline

def ldaLearn(X,y):
    
    #print uniqueClasses
    
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # First we get the number of true classes:

    uniqueClasses = np.unique(y)
    
    means_matrix = []
    for group in uniqueClasses:
        #print y.flatten() == group
        Xg = X[y.flatten() == group, :]       
        #print np.asarray(Xg.mean(0)).shape
        means_matrix.append(Xg.mean(0))
        #print means_temp
    
    #print means_matrix
    
    means=np.transpose(np.asarray(means_matrix))
    covmat = np.cov(X.transpose())
    
    return means,covmat
    
def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    
    uniqueClasses = np.unique(y)
    means_matrix = []
    #covmat_matrix = []
    covmats = [np.zeros((X.shape[1],X.shape[1]))] * uniqueClasses.size
    i=0;
    for group in uniqueClasses:
        
        Xg = X[y.flatten() == group, :]
        means_matrix.append(Xg.mean(0))
        
        Yg = Xg.T
        covmats[i] = np.cov(Yg)
        i=i+1
        #covmat_matrix.append(np.cov(Yg))
    
    means=np.transpose(np.asarray(means_matrix))
    
    return means,covmats
    
def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    
    sigma = linalg.det(covmat)
    inverse_sigma = linalg.inv(covmat)
    denom = np.sqrt(2 * np.pi) * np.square(sigma)
    
    classes = means.shape[1]
    
    total = np.zeros((Xtest.shape[0],classes))
    #print total
    for i in range(classes):
        xMinusU = Xtest - means[:,i]
        total[:,i] = np.exp(-0.5*np.sum(xMinusU * np.dot(inverse_sigma, xMinusU.T).T,1))/denom
    
    #print total
    label = np.argmax(total,1)
    #print label
    label = label + 1
    #print "----"
    #print label
    ytest = ytest.reshape(ytest.size)
    acc = 100*np.mean(label == ytest)
    return acc, label

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD

    classes = means.shape[1]
    total = np.zeros((Xtest.shape[0],classes))
    
    for i in range(classes):
        sigma = linalg.det(covmats[i])
        inverse_sigma = linalg.inv(covmats[i])
        denom = np.sqrt(2 * np.pi) * np.square(sigma)
        xMinusU = Xtest - means[:,i]
        total[:,i] = np.exp(-0.5*np.sum(xMinusU * np.dot(inverse_sigma, xMinusU.T).T,1))/denom
    
    label = np.argmax(total,1)
    #print label
    label = label + 1
    #print "----"
    #print label
    ytest = ytest.reshape(ytest.size)
    acc = 100*np.mean(label == ytest)
    return acc, label
    
def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD 
    a1 = np.dot(X.T,X)
    a2 = np.dot(X.T,y)
    w1 = np.linalg.inv(a1)
    w = np.dot(w1, a2)  
    return w;
    
def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD 
    
    #Identity Matrix
    I = np.eye(X.shape[1],dtype=int) #*X.shape[0]
    
    var1 = np.dot(X.transpose(), X)
    var2 = np.dot(lambd,I)
    var3 = np.dot(X.transpose(), y)
    
    var = np.linalg.inv(var1 + var2)
    
    w = np.dot(var,var3)
    
    return w
    
def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    
    # IMPLEMENT THIS METHOD
    wt = w.reshape((w.shape[0],1))
    #print wt
    rmse = np.sqrt((1.0/Xtest.shape[0])*np.sum(np.square((ytest-np.dot(Xtest,w)))))
    return rmse
    
def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD  
    w = w.reshape(65,1)
    # Formuls: error = 0.5*((y - w*X).T * (y - W)) + 0.5*lambd(w.T*w)
    err_var1 = y - np.dot(X,w)
    err_var2 = 0.5*lambd*np.dot(w.transpose(),w)
    
    error = 0.5*np.dot(err_var1.transpose(),err_var1) + err_var2
    
    # Formula: error_grad = (X.T*X)W - X.T*y + lambd*w
    
    err_grad_var1 = np.dot(np.dot(X.transpose(),X), w)
    err_grad_var2 = np.dot(X.transpose(),y)
    err_grad_var3 = lambd*w
    
    error_grad = (err_grad_var1 - err_grad_var2) + err_grad_var3
    error_grad = error_grad.flatten()
    
    return error, error_grad
    
def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD
    Xd = np.ones((x.shape[0], p + 1))
    for i in range(1, p + 1):
        Xd[:, i] = x ** i
    return Xd       
   # print 'Xd s is ' +str(Xd.shape)
    return Xd
    
# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

plt.show()

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

# Problem 2

if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE for test data without intercept '+str(mle))
print('RMSE for test data with intercept '+str(mle_i))


w = learnOLERegression(X,y)
mle_t = testOLERegression(w,X,y)

w_i = learnOLERegression(X_i,y)
mle_ti = testOLERegression(w_i,X_i,y)

print('RMSE for training data without intercept '+str(mle_t))
print('RMSE for training data with intercept '+str(mle_ti))

# # Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses3 = np.zeros((k,1))
rmses3_train = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    rmses3_train[i] = testOLERegression(w_l,X_i,y)
    i = i + 1

plt.plot(lambdas,rmses3)
plt.plot(lambdas,rmses3_train)
plt.legend(('Testing','Training'))
plt.show()

#Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='BFGS', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    rmses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
    
plt.plot(lambdas,rmses4)
plt.show()

# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))
    
