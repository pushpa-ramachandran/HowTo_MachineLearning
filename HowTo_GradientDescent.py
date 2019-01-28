import numpy as np

def simple_gradient_descent(x,y):
    #find the number of input values
    num_samples = len(x)

    #hyper paramters
    #learning rate  - set a value 0.1 and run it for few iterations. Check the cost function whether it is increasing or decreasing.
    # Change later to 0.05 or 0.07 or 0.001 based on when the cost is reducing across multiple iterations
    learning_rate = 0.01

    #initially set the iterations to a low value (5 or 10) and fix the learning rate. And then increase to 1000 or 10000 till you get the minimum cost
    iterations = 10000

    # initiate slope m and bias b values for the equation y = mx +c
    m =0; b =0;

    for iter in range(iterations):
        #predict the output y_pred
        y_pred = m * x + b

        #find the cost by squaring the difference between actual and predicted output values
        #and adding them for all the samples, and then taking the average
        cost_value = (1/ num_samples) * sum([diff ** 2 for diff in  (y - y_pred)])

        #find the partial derivatives of m and b
        m_derivative = -(2/num_samples) * sum(x * (y - y_pred))
        b_derivative = -(2/num_samples) * (sum (y-y_pred))

        #update m and b with the derivatives calculated
        m = m - (learning_rate * m_derivative)
        b = b - (learning_rate * b_derivative)

        #print the values of m ,b and cost for all the iterations to check the changes
        print ("m {} , b {} , cost {} iteration {} ".format(m, b, cost_value, iter) )

# create a function 'simple_gradient_descent' with parameters for the input(x) and expected output values(y)
x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

#Call 'simple_gradient_descent' and pass the values x and y
simple_gradient_descent(x,y)
