import numpy as np
import matplotlib.pyplot as plt


class layer:
    
    def __init__(self,n_input,n_output,activation):
        
        self.Weights = np.random.random((n_input,n_output)) - 0.5
        self.bias = np.random.random(n_output)  - 1.0
        self.activation = activation
        self.error = None
        self.lastDelta = None
        self.delta = None   
        self.deltaW = None
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
            X = layer.activate(X)
        return X
        
    def backpropagation(self,X,Y,learningRate,a,batch_size):
        

        last_output = self.calculate_forward(X)
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            o_i = np.atleast_2d(X if i == 0 else self.layers[i - 1].output)
            if layer == self.layers[-1]:
                layer.error = Y - last_output
                layer.delta = layer.error * layer.apply_derivative(layer.output)
                # layer.deltaW = layer.delta * o_i.T * learningRate
                layer.deltaW = np.dot(o_i.T,layer.delta) * learningRate 
            else:
                next_layer = self.layers[i + 1]
                layer.error = np.dot(next_layer.delta,next_layer.Weights.T)
                layer.delta = layer.error * layer.apply_derivative(layer.output)
                # layer.deltaW = layer.delta * o_i.T * learningRate
                layer.deltaW = np.dot(o_i.T,layer.delta) * learningRate
                
        self.update(learningRate,X,a,batch_size)
            
    def update(self,learningRate,X,a,batch_size):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if layer.lastDelta is None:
                layer.lastDelta = np.zeros(layer.deltaW.shape)
            layer.Weights += layer.deltaW + a * layer.lastDelta
            layer.lastDelta = layer.deltaW
            # bias = np.matmul(np.ones((1,batch_size)),layer.delta)
            # layer.bias += bias * learningRate
            
    def set_batch(self,X_train,y_train,batch_size):
        batch = np.zeros((batch_size,X_train.shape[2]))
        target = np.zeros((batch_size,12))
        for k in range(batch_size):
            j = np.random.randint(12)
            i = np.random.randint(520)
            batch[k] = X_train[j][i]
            target[k] = y_train[j][i]
        return batch,target
    
    def train(self, X_train, Y_train, learningRate, Iteration,batch_size,a):
        
        
        for i in range(Iteration):  
            # for k in range(batch_size):
            #     j = np.random.randint(12)
            #     pos = np.random.randint(1,520)     
            #     self.backpropagation(X_train[j][pos],Y_train[j][pos],learningRate,a)
            batch, target = self.set_batch(X_train,Y_train,batch_size)
            self.backpropagation(batch,target,learningRate,a,batch_size)
            
 
            print("第%s次迭代" % (i))
                
    def test(self,X_test,Y_test,n):
        temp = 0
        for i in range(n):
            j = np.random.randint(240)
            k = np.random.randint(12)
            output = self.calculate_forward(X_test[k][j])
            index = np.argmax(output)
            if index == k :
                temp +=1
        return temp / n 
           
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
            
    def test_all(self,X_test,Y_test):
        temp = 0
        count = 0
        for i in range(12):
            for j in range(240):
                count += 1
                output = self.calculate_forward(X_test[i][j])
                index = np.argmax(output)
                if index == i :
                    temp +=1
        return temp / count

            
def CreateNetwork(n_cell,act):
    Ne = NeuralNetwork()
    for i in range(len(n_cell) - 1):
        Ne.add_layer(layer(n_cell[i],n_cell[i + 1],act))
    return Ne
    

