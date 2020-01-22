import sys 
import json
import numpy as np

# Use files from command line 
input_file = sys.argv[1]
learning_file = sys.argv[2]

#Read json data values to find learning rate
#and iteration count 
with open(learning_file) as json_file:
    read_json = json.load(json_file)
    learning_rate = read_json['learning rate']
    num_iter = read_json['num iter']

#Load input values as an array, then seperate X1..Xn; Y
arr_all = np.loadtxt(input_file)
X = np.array(arr_all[:,:-1])
Y = np.array(arr_all[:,-1])

#To do the dot product, we need a column of 1s added to make the
# first weight Wo 
col_1s = np.ones((np.shape(X)[0], 1))
X = np.hstack((col_1s, X))

#Next, the weights will be initialized randomly. 
#We have number of Xs + 1 weights (it all comes from y = w1 + w2x1 + w3x2 
# + ... wNxn-1)

#Array to hold all weights 
weights = np.zeros((1,np.shape(X)[1]))

for i in range(np.shape(X)[1]):
    weights[0,i] = np.random.uniform(0,1)
# Analytic Solution
def analytic(x,y):
    X_t = x.transpose()
    X_tX = np.dot(X_t, x)
    X_tX_inv = np.linalg.inv(X_tX)
    X_tX_invX_t = np.dot(X_tX_inv, X_t)
    W_A = np.dot(X_tX_invX_t, y)
    return W_A

# Stochastic Solution
def stochasticGD(x,y,weights,num_its,Lrate):
    for i in range(num_its):
        
        #Select random row from data to start with 
        index = np.random.randint(0, np.shape(x)[0])
        #Compute the changed weight 
        w_new = weights + Lrate*(y[index] - np.dot(weights, x[index]))*x[index]
        # set new weight equal to old weights and loop again
        weights = w_new  
    return weights

# To double check working solution, uncomment
#print(stochasticGD(X,Y,weights,num_iter,learning_rate))
#print(analytic(X,Y))

SGD_weight = stochasticGD(X,Y,weights,num_iter,learning_rate)

SGD_weights = SGD_weight[0]
analytic_weights = analytic(X,Y)

#Writing into an 'out' file
def return_output(analytic, SGD):
      input_name = input_file.split('.')[0]
      output_name = f"{input_name}.out"
      with open(output_name, 'w') as f:
         for weight in analytic:
             f.write(f"{weight:.4f}\n")
         f.write("\n")
         for weight in SGD:
             f.write(f"{weight:.4f}\n")

#Final output
return_output(analytic_weights, SGD_weights);