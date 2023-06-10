import CurveFittingSin as CFS
import numpy as np
import matplotlib.pyplot as plt


x_train = np.random.random((100,1)) * np.pi * 2 - np.pi
y_train = np.sin(x_train) 
x_test = np.random.random((100,1)) * 2 * np.pi - np.pi
y_test = np.sin(x_test)

act = "tanh"
Ne = CFS.CreateNetwork((1,8,16,32,16,8,1),act)
Ne.train(x_train,y_train,0.05,2000)
error = Ne.test(x_test,y_test)
print("测试集误差为：%.6f" % (error))

# Ne.print_weight()
x_darw = np.zeros((100,1))
for i in range(100):
    temp = np.linspace(-np.pi + i * np.pi/50 ,-np.pi +  (i + 1) * np.pi/50 , 1)
    x_darw[i] = temp
x_train = x_train.flatten()
y_train = y_train.flatten()
plt.scatter(x_train,y_train, color = 'r')    
Ne.visual_data(x_darw,100)
plt.show()