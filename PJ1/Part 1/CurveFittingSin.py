import numpy as np
import matplotlib.pyplot as plt


class layer:
    
    def __init__(self,n_input,n_output,activation):
        
        self.Weights = np.random.random((n_input,n_output)) - 0.5
        self.bias = np.random.rand(n_output)  - 1.0
        self.activation = activation
        self.error = None
        self.delta = None   
        self.output = None

        
    def activate(self,X):
        
        input = np.matmul(X,self.Weights) + self.bias   
        self.output = self.apply_activate(input)
        return self.output
    
    def apply_activate(self,X):
        
        if self.activation == 'relu':
            return np.maximum(X,0)
        elif self.activation == 'tanh':
            return np.tanh(X)
        elif self.activation == 'sigmod':
            return (1.0 / (1.0 + np.exp(-X)))
    
    def apply_derivative(self,X):
        
        if self.activation == 'relu':
            grad = np.array(X,copy=True)
            grad[X > 0] = 1
            grad[X <= 0] = 0
            return grad
        elif self.activation == 'tanh':
            return 1 - X ** 2
        elif self.activation == 'sigmod':
            return X * (1 - X)
    

class NeuralNetwork:
    
    def __init__(self):
        self.layers = []
        self.n_layer = 0
        
    def add_layer(self,layer):
        self.layers.append(layer)
        self.n_layer += 1
        
    def calculate_forward(self,X):
        for i in range(self.n_layer):
            layer = self.layers[i]
            if layer != self.layers[-1]:
                X = layer.activate(X)
            else:
                X = layer.activate(X)
        return X
        
    def backpropagation(self,X,Y,learningRate):
        
        # last_output = self.calculate_forward(X)
        # print(last_output)
        
        last_output = self.calculate_forward(X)
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if layer == self.layers[-1]:
                layer.error = Y - last_output
                layer.delta = layer.error * layer.apply_derivative(layer.output)
            else:
                next_layer = self.layers[i + 1]
                layer.error = np.dot(next_layer.Weights, next_layer.delta)
                # print(layer.error)
                # print(layer.apply_derivative(layer.output))
                layer.delta = layer.error * layer.apply_derivative(layer.output)
                
        self.update(learningRate,X)
            
    def update(self,learningRate,X):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            o_i = np.atleast_2d(X if i == 0 else self.layers[i - 1].output)
            layer.Weights += layer.delta * o_i.T * learningRate
    
    def train(self, X_train, Y_train, learningRate, Iteration):
        
        
        for i in range(Iteration):  
            for j in range(len(X_train)): 
                self.backpropagation(X_train[j],Y_train[j],learningRate)
 
                # self.backpropagation(X_train,Y_train,learningRate)
            print("第%s次迭代,误差为：%.6f" % (i,np.abs(np.mean(Y_train[j] - self.layers[self.n_layer - 1].output))))
                
    def test(self,X_test,Y_test):
        temp = 0
        for i in range(len(X_test)):
            error = Y_test[i] - self.calculate_forward(X_test[i])
            temp += np.abs(np.mean(error))
        return temp / len(X_test)
        
            
        
    def visual_data(self,X,n):
        Y = np.zeros((n,self.layers[0].Weights.shape[0]))
        for i in range(n):
            Y[i] = self.calculate_forward(X[i])   
        X_darw = X.flatten()  
        Y = Y.flatten()
        plt.scatter(X_darw,Y,color = 'y')
    
    
    def print_weight(self):
        for layer in self.layers:
            print(layer.Weights) 

def CreateNetwork(n_cell,act):
    Ne = NeuralNetwork()
    for i in range(len(n_cell) - 1):
        Ne.add_layer(layer(n_cell[i],n_cell[i + 1],act))
    return Ne

