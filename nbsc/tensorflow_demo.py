import tensorflow as tf # Needs python 3.5.2

# Standard imports
import numpy
import urllib
import random
from math import exp
from math import log

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

# Read the data (as in hw2)
dataFile = open("C:/Users/julia/Downloads/winequality-white.csv")
header = dataFile.readline()
fields = ["constant"] + header.strip().replace('"','').split(';')
featureNames = fields[:-1]
labelName = fields[-1]
lines = [[1.0] + [float(x) for x in l.split(';')] for l in dataFile]
X = [l[:-1] for l in lines]
y_reg = [l[-1] for l in lines] # For regression
y_class = [1.0*(l[-1] > 5) for l in lines] # for classification

y_reg = tf.constant(y_reg, shape=[len(y_reg),1])
y_class = tf.constant(y_class, shape=[len(y_class),1])


# Standard MSE in tensorflow
def MSE(X, y, theta):
  return tf.reduce_mean((tf.matmul(X,theta) - y)**2)

# Mean-squared error with a regularizer
def MSE_regularized(X, y, theta, lamb):
  return tf.reduce_mean((tf.matmul(X,theta) - y)**2)\
  + lamb*tf.reduce_sum(theta**2)

# Logistic regression
def logreg(X, y, theta):
  error1 = tf.multiply(y, tf.log(tf.sigmoid(tf.matmul(X,theta))))
  error2 = tf.multiply((1-y), tf.log(1 - tf.sigmoid(tf.matmul(X,theta))))
  return tf.reduce_sum(error1) + tf.reduce_sum(error2)


# Create a new variable, and initialize to a vector of zeros
theta = tf.Variable(tf.constant([0.0]*len(featureNames), shape=[len(featureNames),1]))

# Stochastic gradient descent
optimizer = tf.train.AdamOptimizer(0.01)

# The objective we'll optimize is the MSE
objective = MSE(X,y_reg,theta)

# Our goal is to minimize it
train = optimizer.minimize(objective)

# Initialize our variables
init = tf.global_variables_initializer()

# Create a new optimization session
sess = tf.Session()
sess.run(init)

# Run 20 iterations of gradient descent
for iteration in range(20):
  cvalues = sess.run([train, objective])
  print("objective = " + str(cvalues[1]))

# Print the outputs
with sess.as_default():
  print(MSE(X, y_reg, theta).eval())
  print(theta.eval())
